// Copyright (c) 2024 Hugo Chiang
// SPDX-License-Identifier: MIT

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "vqf.h"
#include <IQmath_RV32.h>

#define sqrt_fast sqrtf
#define fabs_fast fabsf
#define exp_fast expf

#define EPS FLT_EPSILON
#define NaN NAN
#define M_SQRT2f     1.41421356237309504880f   // sqrt(2)s
#define M_PIf       3.14159265358979323846f
#define M_SQRT2d     1.41421356237309504880168872420969808
#define M_PId       3.14159265358979323846264338327950288

typedef struct vqf_params_s {
    vqf_real_t tauAcc;
    vqf_real_t tauMag;
    bool motionBiasEstEnabled;
    bool restBiasEstEnabled;
    bool magDistRejectionEnabled;
    vqf_real_t biasSigmaInit;
    vqf_real_t biasForgettingTime;
    vqf_real_t biasClip;
    vqf_real_t biasSigmaMotion;
    vqf_real_t biasVerticalForgettingFactor;
    vqf_real_t biasSigmaRest;
    vqf_real_t restMinT;
    vqf_real_t restFilterTau;
    vqf_real_t restThGyr;
    vqf_real_t restThAcc;
    vqf_real_t magCurrentTau;
    vqf_real_t magRefTau;
    vqf_real_t magNormTh;
    vqf_real_t magDipTh;
    vqf_real_t magNewTime;
    vqf_real_t magNewFirstTime;
    vqf_real_t magNewMinGyr;
    vqf_real_t magMinUndisturbedTime;
    vqf_real_t magMaxRejectionTime;
    vqf_real_t magRejectionFactor;
} vqf_params_t;

typedef int32_t vqf_q30_t;
typedef int32_t vqf_q24_t;
typedef int32_t vqf_q16_t;

#define VQF_Q30_ONE ((vqf_q30_t)0x40000000)
#define VQF_Q16_ONE ((vqf_q16_t)0x00010000)

static vqf_real_t q30_to_real(vqf_q30_t value)
{
    return (vqf_real_t)value * (vqf_real_t)(1.0 / 1073741824.0);
}

static vqf_q16_t q16_from_real(vqf_real_t value)
{
    if (value >= 32767.9999847f) {
        return INT32_MAX;
    }
    if (value <= -32768.0f) {
        return INT32_MIN;
    }
    return (vqf_q16_t)(value * 65536.0f);
}

static vqf_real_t q16_to_real(vqf_q16_t value)
{
    return (vqf_real_t)value * (vqf_real_t)(1.0 / 65536.0);
}

static vqf_q30_t q30_mul(vqf_q30_t a, vqf_q30_t b)
{
    return (vqf_q30_t)(((int64_t)a * b) >> 30);
}

static vqf_q30_t q30_div(vqf_q30_t a, vqf_q30_t b)
{
    return (vqf_q30_t)_IQ30div_FAST(a, b);
}

static vqf_q16_t q16_div(vqf_q16_t a, vqf_q16_t b)
{
    return (vqf_q16_t)_IQ16div_FAST(a, b);
}

static vqf_q16_t q16_norm(const vqf_q16_t vec[], size_t n)
{
    int64_t sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += (int64_t)vec[i] * vec[i];
    }
    if (sum == 0) {
        return 0;
    }
    sum >>= 16;
    if (sum > INT32_MAX) {
        sum = INT32_MAX;
    }
    return (vqf_q16_t)_IQ16sqrt((vqf_q16_t)sum);
}

static vqf_q30_t q30_norm(const vqf_q30_t vec[], size_t n)
{
    int64_t sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += (int64_t)vec[i] * vec[i];
    }
    if (sum == 0) {
        return 0;
    }
    if ((uint64_t)sum >> 30 > INT32_MAX) {
        sum = (int64_t)INT32_MAX << 30;
    }
    return (vqf_q30_t)_IQ30sqrt((vqf_q30_t)(sum >> 30));
}

static vqf_q30_t q30_from_i64(int64_t value)
{
    if (value > INT32_MAX) {
        return INT32_MAX;
    }
    if (value < INT32_MIN) {
        return INT32_MIN;
    }
    return (vqf_q30_t)value;
}

static vqf_q24_t q24_clamp_unit(vqf_q24_t value)
{
    const vqf_q24_t one = (vqf_q24_t)0x01000000;
    if (value > one) {
        return one;
    }
    if (value < -one) {
        return -one;
    }
    return value;
}

static void q30_normalize(vqf_q30_t vec[], size_t n)
{
    vqf_q30_t nrm = q30_norm(vec, n);
    if (nrm == 0) {
        return;
    }
    for (size_t i = 0; i < n; i++) {
        vec[i] = q30_div(vec[i], nrm);
    }
}

static void q16_normalize(vqf_q16_t vec[], size_t n)
{
    vqf_q16_t nrm = q16_norm(vec, n);
    if (nrm == 0) {
        return;
    }
    for (size_t i = 0; i < n; i++) {
        vec[i] = q16_div(vec[i], nrm);
    }
}


typedef struct vqf_coeffs_s {
    vqf_real_t gyrTs;
    vqf_real_t accTs;
    vqf_real_t magTs;
    vqf_double_t accLpB[3];
    vqf_double_t accLpA[2];
    vqf_real_t kMag;
    vqf_real_t biasP0;
    vqf_real_t biasV;
    vqf_real_t biasMotionW;
    vqf_real_t biasVerticalW;
    vqf_real_t biasRestW;
    vqf_double_t restGyrLpB[3];
    vqf_double_t restGyrLpA[2];
    vqf_double_t restAccLpB[3];
    vqf_double_t restAccLpA[2];
    vqf_real_t kMagRef;
    vqf_double_t magNormDipLpB[3];
    vqf_double_t magNormDipLpA[2];
} vqf_coeffs_t ;

typedef struct vqf_state_s {
    vqf_q30_t gyrQuat[4];
    vqf_q30_t accQuat[4];
    vqf_real_t delta;
    bool restDetected;
    bool magDistDetected;
    vqf_real_t lastAccLp[3];
    vqf_double_t accLpState[3*2];
    vqf_real_t lastAccCorrAngularRate;
    vqf_real_t kMagInit;
    vqf_real_t lastMagDisAngle;
    vqf_real_t lastMagCorrAngularRate;
    vqf_real_t bias[3];
    vqf_q16_t biasQ16[3];
    vqf_real_t biasP[9];
    vqf_double_t motionBiasEstRLpState[9*2];
    vqf_double_t motionBiasEstBiasLpState[2*2];
    vqf_real_t restLastSquaredDeviations[2];
    vqf_real_t restT;
    vqf_real_t restLastGyrLp[3];
    vqf_double_t restGyrLpState[3*2];
    vqf_real_t restLastAccLp[3];
    vqf_double_t restAccLpState[3*2];
    vqf_real_t magRefNorm;
    vqf_real_t magRefDip;
    vqf_real_t magUndisturbedT;
    vqf_real_t magRejectT;
    vqf_real_t magCandidateNorm;
    vqf_real_t magCandidateDip;
    vqf_real_t magCandidateT;
    vqf_real_t magNormDip[2];
    vqf_double_t magNormDipLpState[2*2];
} vqf_state_t;

static vqf_params_t params;
static vqf_coeffs_t coeffs;
static vqf_state_t state;

static void init_params(void)
{
    params.tauAcc = 3.0f;
    params.tauMag = 9.0f;
    params.motionBiasEstEnabled = true;
    params.restBiasEstEnabled = true;
    params.magDistRejectionEnabled = true;
    params.biasSigmaInit = 0.5f;
    params.biasForgettingTime = 100.0f;
    params.biasClip = 2.0f;
    params.biasSigmaMotion = 0.1f;
    params.biasVerticalForgettingFactor = 0.0001f;
    params.biasSigmaRest = 0.03f;
    params.restMinT = 1.5f;
    params.restFilterTau = 0.5f;
    params.restThGyr = 2.0f;
    params.restThAcc = 0.5f;
    params.magCurrentTau = 0.05f;
    params.magRefTau = 20.0f;
    params.magNormTh = 0.1f;
    params.magDipTh = 10.0f;
    params.magNewTime = 20.0f;
    params.magNewFirstTime = 5.0f;
    params.magNewMinGyr = 20.0f;
    params.magMinUndisturbedTime = 0.5f;
    params.magMaxRejectionTime = 60.0f;
    params.magRejectionFactor = 2.0f;
}

static void vqf_fill_real(vqf_real_t *dst, size_t n, vqf_real_t val)
{
    for (size_t i = 0; i < n; i++) {
        dst[i] = val;
    }
}

static void vqf_fill_double(vqf_double_t *dst, size_t n, vqf_double_t val)
{
    for (size_t i = 0; i < n; i++) {
        dst[i] = val;
    }
}
static float vqf_square(float x) {
    return x*x;
}

static float vqf_max(float a, float b) {
    return a > b ? a : b;
}

static float vqf_min(float a, float b) {
    return a < b ? a : b;
}

static vqf_real_t norm(const vqf_real_t vec[], size_t N)
{
    vqf_real_t s = 0;
    for(size_t i = 0; i < N; i++) {
        s += vec[i]*vec[i];
    }
    // sqrt can be replaced by arm_sqrt_f32 from CMSIS_DSP
    return sqrt_fast(s);
}

static void normalize(vqf_real_t vec[], size_t N)
{
    vqf_real_t n = norm(vec, N);
    if (n < EPS) {
        return;
    }
    for(size_t i = 0; i < N; i++) {
        vec[i] /= n;
    }
}

static void clip(vqf_real_t vec[], size_t N, vqf_real_t min, vqf_real_t max)
{
    for(size_t i = 0; i < N; i++) {
        if (vec[i] < min) {
            vec[i] = min;
        } else if (vec[i] > max) {
            vec[i] = max;
        }
    }
}


static void quatMultiply(const vqf_q30_t q1[4], const vqf_q30_t q2[4], vqf_q30_t out[4])
{
    int64_t w = (int64_t)q1[0] * q2[0] - (int64_t)q1[1] * q2[1]
            - (int64_t)q1[2] * q2[2] - (int64_t)q1[3] * q2[3];
    int64_t x = (int64_t)q1[0] * q2[1] + (int64_t)q1[1] * q2[0]
            + (int64_t)q1[2] * q2[3] - (int64_t)q1[3] * q2[2];
    int64_t y = (int64_t)q1[0] * q2[2] - (int64_t)q1[1] * q2[3]
            + (int64_t)q1[2] * q2[0] + (int64_t)q1[3] * q2[1];
    int64_t z = (int64_t)q1[0] * q2[3] + (int64_t)q1[1] * q2[2]
            - (int64_t)q1[2] * q2[1] + (int64_t)q1[3] * q2[0];
    out[0] = q30_from_i64(w >> 30);
    out[1] = q30_from_i64(x >> 30);
    out[2] = q30_from_i64(y >> 30);
    out[3] = q30_from_i64(z >> 30);
}

static void quatToMatrix(const vqf_q30_t q[4], vqf_q30_t out[9]);

static void quatConj(const vqf_q30_t q[4], vqf_q30_t out[4])
{
    vqf_q30_t w = q[0];
    vqf_q30_t x = -q[1];
    vqf_q30_t y = -q[2];
    vqf_q30_t z = -q[3];
    out[0] = w; out[1] = x; out[2] = y; out[3] = z;
}


static void quatSetToIdentity(vqf_q30_t out[4])
{
    out[0] = VQF_Q30_ONE;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
}

static void quatApplyDelta(vqf_q30_t q[], vqf_real_t delta, vqf_q30_t out[])
{
    // out = quatMultiply([cos(delta/2), 0, 0, sin(delta/2)], q)
    vqf_q24_t angle = (vqf_q24_t)_IQ24(delta * (vqf_real_t)0.5);
    vqf_q30_t c = (vqf_q30_t)((int64_t)_IQ24cos(angle) * 64);
    vqf_q30_t s = (vqf_q30_t)((int64_t)_IQ24sin(angle) * 64);
    vqf_q30_t w = q30_mul(c, q[0]) - q30_mul(s, q[3]);
    vqf_q30_t x = q30_mul(c, q[1]) - q30_mul(s, q[2]);
    vqf_q30_t y = q30_mul(c, q[2]) + q30_mul(s, q[1]);
    vqf_q30_t z = q30_mul(c, q[3]) + q30_mul(s, q[0]);
    out[0] = w; out[1] = x; out[2] = y; out[3] = z;
}

static void quatRotate(const vqf_q30_t q[4], const vqf_q16_t v[3], vqf_q16_t out[3])
{
    vqf_q30_t r[9];
    quatToMatrix(q, r);
    for (size_t i = 0; i < 3; i++) {
        int64_t value = (int64_t)r[3*i] * v[0] + (int64_t)r[3*i+1] * v[1]
                + (int64_t)r[3*i+2] * v[2];
        out[i] = (vqf_q16_t)(value >> 30);
    }
}

static void quatToMatrix(const vqf_q30_t q[4], vqf_q30_t out[9])
{
    vqf_q30_t products[9];
    products[0] = q30_mul(q[2], q[2]);
    products[1] = q30_mul(q[3], q[3]);
    products[2] = q30_mul(q[2], q[1]);
    products[3] = q30_mul(q[0], q[3]);
    products[4] = q30_mul(q[0], q[2]);
    products[5] = q30_mul(q[3], q[1]);
    products[6] = q30_mul(q[1], q[1]);
    products[7] = q30_mul(q[2], q[3]);
    products[8] = q30_mul(q[1], q[0]);
    out[0] = q30_from_i64((int64_t)VQF_Q30_ONE - 2 * (int64_t)products[0] - 2 * (int64_t)products[1]);
    out[1] = q30_from_i64(2 * ((int64_t)products[2] - products[3]));
    out[2] = q30_from_i64(2 * ((int64_t)products[4] + products[5]));
    out[3] = q30_from_i64(2 * ((int64_t)products[3] + products[2]));
    out[4] = q30_from_i64((int64_t)VQF_Q30_ONE - 2 * (int64_t)products[6] - 2 * (int64_t)products[1]);
    out[5] = q30_from_i64(2 * ((int64_t)products[7] - products[8]));
    out[6] = q30_from_i64(2 * ((int64_t)products[5] - products[4]));
    out[7] = q30_from_i64(2 * ((int64_t)products[8] + products[7]));
    out[8] = q30_from_i64((int64_t)VQF_Q30_ONE - 2 * (int64_t)products[6]
            - 2 * (int64_t)products[0]);
}




static vqf_real_t gainFromTau(vqf_real_t tau, vqf_real_t Ts)
{
    // assert(Ts > 0);
    if (tau < (vqf_real_t)(0.0)) {
        return 0; // k=0 for negative tau (disable update)
    } else if (tau == (vqf_real_t)(0.0)) {
        return 1; // k=1 for tau=0
    } else {
        return 1 - exp_fast(-Ts/tau);  // fc = 1/(2*pi*tau)
    }
}

static void filterCoeffs(vqf_real_t tau, vqf_real_t Ts, vqf_double_t outB[], vqf_double_t outA[])
{
    // assert(tau > 0);
    // assert(Ts > 0);
    // second order Butterworth filter based on https://stackoverflow.com/a/52764064
    vqf_double_t fc = (M_SQRT2d / (2.0*M_PId))/(vqf_double_t)(tau); // time constant of dampened, non-oscillating part of step response
    vqf_double_t C = tan(M_PId*fc*(vqf_double_t)(Ts));
    // sqrt can be replaced by arm_sqrt_f32 from CMSIS_DSP
    vqf_double_t D = C*C + M_SQRT2d*C + 1;
    vqf_double_t b0 = C*C/D;
    outB[0] = b0;
    outB[1] = 2*b0;
    outB[2] = b0;
    // a0 = 1.0
    outA[0] = 2*(C*C-1)/D; // a1
    // sqrt can be replaced by arm_sqrt_f32 from CMSIS_DSP
    outA[1] = (1-M_SQRT2d*C+C*C)/D; // a2
}

static void filterInitialState(vqf_real_t x0, const vqf_double_t b[3], const vqf_double_t a[2], vqf_double_t out[])
{
    // initial state for steady state (equivalent to scipy.signal.lfilter_zi, obtained by setting y=x=x0 in the filter
    // update equation)
    out[0] = x0*(1 - b[0]);
    out[1] = x0*(b[2] - a[1]);
}

static void filterAdaptStateForCoeffChange(vqf_real_t last_y[], size_t N, const vqf_double_t b_old[],
                                         const vqf_double_t a_old[], const vqf_double_t b_new[],
                                         const vqf_double_t a_new[], vqf_double_t state[])
{
    if (isnan(state[0])) {
        return;
    }
    for (size_t i = 0; i < N; i++) {
        state[0+2*i] = state[0+2*i] + (b_old[0] - b_new[0])*last_y[i];
        state[1+2*i] = state[1+2*i] + (b_old[1] - b_new[1] - a_old[0] + a_new[0])*last_y[i];
    }
}

static vqf_real_t filterStep(vqf_real_t x, const vqf_double_t b[3], const vqf_double_t a[2], vqf_double_t state[2])
{
    // difference equations based on scipy.signal.lfilter documentation
    // assumes that a0 == 1.0
    vqf_double_t y = b[0]*x + state[0];
    state[0] = b[1]*x - a[0]*y + state[1];
    state[1] = b[2]*x - a[1]*y;
    return (vqf_real_t)y;
}

static void filterVec(const vqf_real_t x[], size_t N, vqf_real_t tau, vqf_real_t Ts, const vqf_double_t b[3],
                    const vqf_double_t a[2], vqf_double_t state[], vqf_real_t out[])
{
    // assert(N>=2);

    // to avoid depending on a single sample, average the first samples (for duration tau)
    // and then use this average to calculate the filter initial state
    if (isnan(state[0])) { // initialization phase
        if (isnan(state[1])) { // first sample
            state[1] = 0; // state[1] is used to store the sample count
            for(size_t i = 0; i < N; i++) {
                state[2+i] = 0; // state[2+i] is used to store the sum
            }
        }
        state[1]++;
        for (size_t i = 0; i < N; i++) {
            state[2+i] += x[i];
            out[i] = (vqf_real_t)(state[2+i]/state[1]);
        }
        if (state[1]*Ts >= tau) {
            for(size_t i = 0; i < N; i++) {
               filterInitialState(out[i], b, a, state+2*i);
            }
        }
        return;
    }

    for (size_t i = 0; i < N; i++) {
        out[i] = filterStep(x[i], b, a, state+2*i);
    }
}

static void matrix3SetToScaledIdentity(vqf_real_t scale, vqf_real_t out[9])
{
    out[0] = scale;
    out[1] = 0.0;
    out[2] = 0.0;
    out[3] = 0.0;
    out[4] = scale;
    out[5] = 0.0;
    out[6] = 0.0;
    out[7] = 0.0;
    out[8] = scale;
}

static void matrix3Multiply(const vqf_real_t in1[9], const vqf_real_t in2[9], vqf_real_t out[9])
{
    vqf_real_t tmp[9];
    tmp[0] = in1[0]*in2[0] + in1[1]*in2[3] + in1[2]*in2[6];
    tmp[1] = in1[0]*in2[1] + in1[1]*in2[4] + in1[2]*in2[7];
    tmp[2] = in1[0]*in2[2] + in1[1]*in2[5] + in1[2]*in2[8];
    tmp[3] = in1[3]*in2[0] + in1[4]*in2[3] + in1[5]*in2[6];
    tmp[4] = in1[3]*in2[1] + in1[4]*in2[4] + in1[5]*in2[7];
    tmp[5] = in1[3]*in2[2] + in1[4]*in2[5] + in1[5]*in2[8];
    tmp[6] = in1[6]*in2[0] + in1[7]*in2[3] + in1[8]*in2[6];
    tmp[7] = in1[6]*in2[1] + in1[7]*in2[4] + in1[8]*in2[7];
    tmp[8] = in1[6]*in2[2] + in1[7]*in2[5] + in1[8]*in2[8];
    memcpy(out, tmp, sizeof(tmp));
    // std::copy(tmp, tmp+9, out);
}

static void matrix3MultiplyTpsFirst(const vqf_real_t in1[9], const vqf_real_t in2[9], vqf_real_t out[9])
{
    vqf_real_t tmp[9];
    tmp[0] = in1[0]*in2[0] + in1[3]*in2[3] + in1[6]*in2[6];
    tmp[1] = in1[0]*in2[1] + in1[3]*in2[4] + in1[6]*in2[7];
    tmp[2] = in1[0]*in2[2] + in1[3]*in2[5] + in1[6]*in2[8];
    tmp[3] = in1[1]*in2[0] + in1[4]*in2[3] + in1[7]*in2[6];
    tmp[4] = in1[1]*in2[1] + in1[4]*in2[4] + in1[7]*in2[7];
    tmp[5] = in1[1]*in2[2] + in1[4]*in2[5] + in1[7]*in2[8];
    tmp[6] = in1[2]*in2[0] + in1[5]*in2[3] + in1[8]*in2[6];
    tmp[7] = in1[2]*in2[1] + in1[5]*in2[4] + in1[8]*in2[7];
    tmp[8] = in1[2]*in2[2] + in1[5]*in2[5] + in1[8]*in2[8];
    memcpy(out, tmp, sizeof(tmp));
    // std::copy(tmp, tmp+9, out);
}

static void matrix3MultiplyTpsSecond(const vqf_real_t in1[9], const vqf_real_t in2[9], vqf_real_t out[9])
{
    vqf_real_t tmp[9];
    tmp[0] = in1[0]*in2[0] + in1[1]*in2[1] + in1[2]*in2[2];
    tmp[1] = in1[0]*in2[3] + in1[1]*in2[4] + in1[2]*in2[5];
    tmp[2] = in1[0]*in2[6] + in1[1]*in2[7] + in1[2]*in2[8];
    tmp[3] = in1[3]*in2[0] + in1[4]*in2[1] + in1[5]*in2[2];
    tmp[4] = in1[3]*in2[3] + in1[4]*in2[4] + in1[5]*in2[5];
    tmp[5] = in1[3]*in2[6] + in1[4]*in2[7] + in1[5]*in2[8];
    tmp[6] = in1[6]*in2[0] + in1[7]*in2[1] + in1[8]*in2[2];
    tmp[7] = in1[6]*in2[3] + in1[7]*in2[4] + in1[8]*in2[5];
    tmp[8] = in1[6]*in2[6] + in1[7]*in2[7] + in1[8]*in2[8];
    memcpy(out, tmp, sizeof(tmp));
    // std::copy(tmp, tmp+9, out);
}

static bool matrix3Inv(const vqf_real_t in[9], vqf_real_t out[9])
{
    // in = [a b c; d e f; g h i]
    vqf_double_t A = in[4]*in[8] - in[5]*in[7]; // (e*i - f*h)
    vqf_double_t D = in[2]*in[7] - in[1]*in[8]; // -(b*i - c*h)
    vqf_double_t G = in[1]*in[5] - in[2]*in[4]; // (b*f - c*e)
    vqf_double_t B = in[5]*in[6] - in[3]*in[8]; // -(d*i - f*g)
    vqf_double_t E = in[0]*in[8] - in[2]*in[6]; // (a*i - c*g)
    vqf_double_t H = in[2]*in[3] - in[0]*in[5]; // -(a*f - c*d)
    vqf_double_t C = in[3]*in[7] - in[4]*in[6]; // (d*h - e*g)
    vqf_double_t F = in[1]*in[6] - in[0]*in[7]; // -(a*h - b*g)
    vqf_double_t I = in[0]*in[4] - in[1]*in[3]; // (a*e - b*d)

    vqf_double_t det = in[0]*A + in[1]*B + in[2]*C; // a*A + b*B + c*C;

    if (det >= -EPS && det <= EPS) {
        vqf_fill_real(out, 9, 0);
        // std::fill(out, out+9, 0);
        return false;
    }

    // out = [A D G; B E H; C F I]/det
    out[0] = (vqf_real_t)(A/det);
    out[1] = (vqf_real_t)(D/det);
    out[2] = (vqf_real_t)(G/det);
    out[3] = (vqf_real_t)(B/det);
    out[4] = (vqf_real_t)(E/det);
    out[5] = (vqf_real_t)(H/det);
    out[6] = (vqf_real_t)(C/det);
    out[7] = (vqf_real_t)(F/det);
    out[8] = (vqf_real_t)(I/det);

    return true;
}







void updateGyr(const vqf_real_t gyr[3])
{
    // rest detection
    if (params.restBiasEstEnabled || params.magDistRejectionEnabled) {
        filterVec(gyr, 3, params.restFilterTau, coeffs.gyrTs, coeffs.restGyrLpB, coeffs.restGyrLpA,
                  state.restGyrLpState, state.restLastGyrLp);

        state.restLastSquaredDeviations[0] = vqf_square(gyr[0] - state.restLastGyrLp[0])
                + vqf_square(gyr[1] - state.restLastGyrLp[1]) + vqf_square(gyr[2] - state.restLastGyrLp[2]);

        vqf_real_t biasClip = params.biasClip*(vqf_real_t)(M_PIf/180.0);
        if (state.restLastSquaredDeviations[0] >= vqf_square(params.restThGyr*(vqf_real_t)(M_PIf/180.0))
                // fabs_fast can be replaced by arm_abs_f32 from CMSIS-DSP
                || fabs_fast(state.restLastGyrLp[0]) > biasClip || fabs_fast(state.restLastGyrLp[1]) > biasClip
                || fabs_fast(state.restLastGyrLp[2]) > biasClip) {
            state.restT = 0.0;
            state.restDetected = false;
        }
    }

    // Remove bias and integrate in fixed point. The filters above intentionally
    // remain in double precision; only the quaternion prediction crosses here.
    vqf_q16_t gyrNoBias[3] = {
        q16_from_real(gyr[0]) - state.biasQ16[0],
        q16_from_real(gyr[1]) - state.biasQ16[1],
        q16_from_real(gyr[2]) - state.biasQ16[2]
    };
    vqf_q16_t gyrNorm = q16_norm(gyrNoBias, 3);
    if (gyrNorm > 1) {
        vqf_q16_t gyrTs = q16_from_real(coeffs.gyrTs);
        vqf_q24_t angle = (vqf_q24_t)(((int64_t)gyrNorm * gyrTs) >> 8);
        vqf_q24_t halfAngle = angle >> 1;
        vqf_q30_t axis[3];
        vqf_q30_t gyrStepQuat[4];
        vqf_q24_t sine = (vqf_q24_t)_IQ24sin(halfAngle);
        for (size_t i = 0; i < 3; i++) {
            axis[i] = (vqf_q30_t)((int64_t)q16_div(gyrNoBias[i], gyrNorm) * 16384);
        }
        gyrStepQuat[0] = (vqf_q30_t)((int64_t)_IQ24cos(halfAngle) * 64);
        for (size_t i = 0; i < 3; i++) {
            gyrStepQuat[i + 1] = (vqf_q30_t)(((int64_t)sine * axis[i]) >> 24);
        }
        quatMultiply(state.gyrQuat, gyrStepQuat, state.gyrQuat);
        q30_normalize(state.gyrQuat, 4);
    }
}

void updateAcc(const vqf_real_t acc[3])
{
    // ignore [0 0 0] samples
    if (acc[0] == (vqf_real_t)(0.0) && acc[1] == (vqf_real_t)(0.0) && acc[2] == (vqf_real_t)(0.0)) {
        return;
    }

    // rest detection
    if (params.restBiasEstEnabled) {
        filterVec(acc, 3, params.restFilterTau, coeffs.accTs, coeffs.restAccLpB, coeffs.restAccLpA,
                  state.restAccLpState, state.restLastAccLp);

        state.restLastSquaredDeviations[1] = vqf_square(acc[0] - state.restLastAccLp[0])
                + vqf_square(acc[1] - state.restLastAccLp[1]) + vqf_square(acc[2] - state.restLastAccLp[2]);

        if (state.restLastSquaredDeviations[1] >= vqf_square(params.restThAcc)) {
            state.restT = 0.0;
            state.restDetected = false;
        } else {
            state.restT += coeffs.accTs;
            if (state.restT >= params.restMinT) {
                state.restDetected = true;
            }
        }
    }

    vqf_real_t accEarth[3];
    vqf_q16_t accFixed[3];
    vqf_q16_t accEarthFixed[3];

    // filter acc in inertial frame
    for (size_t i = 0; i < 3; i++) {
        accFixed[i] = q16_from_real(acc[i]);
    }
    quatRotate(state.gyrQuat, accFixed, accEarthFixed);
    for (size_t i = 0; i < 3; i++) {
        accEarth[i] = q16_to_real(accEarthFixed[i]);
    }
    filterVec(accEarth, 3, params.tauAcc, coeffs.accTs, coeffs.accLpB, coeffs.accLpA, state.accLpState, state.lastAccLp);

    // transform to 6D earth frame and normalize
    for (size_t i = 0; i < 3; i++) {
        accFixed[i] = q16_from_real(state.lastAccLp[i]);
    }
    quatRotate(state.accQuat, accFixed, accEarthFixed);
    q16_normalize(accEarthFixed, 3);
    for (size_t i = 0; i < 3; i++) {
        accEarth[i] = q16_to_real(accEarthFixed[i]);
    }

    // inclination correction
    vqf_q16_t q_w_sq = (vqf_q16_t)((accEarthFixed[2] + VQF_Q16_ONE) >> 1);
        vqf_q16_t q_w = q_w_sq > 0 ? (vqf_q16_t)_IQ16sqrt(q_w_sq) : 0;
    vqf_q30_t accCorrQuat[4];
    if (q_w > 0) {
        accCorrQuat[0] = (vqf_q30_t)((int64_t)q_w * 16384);
        accCorrQuat[1] = (vqf_q30_t)((int64_t)(q16_div(accEarthFixed[1], q_w) >> 1) * 16384);
        accCorrQuat[2] = (vqf_q30_t)((int64_t)(-q16_div(accEarthFixed[0], q_w) >> 1) * 16384);
        accCorrQuat[3] = 0;
    } else {
        // to avoid numeric issues when acc is close to [0 0 -1], i.e. the correction step is close (<= 0.00011°) to 180°:
        accCorrQuat[0] = 0;
        accCorrQuat[1] = VQF_Q30_ONE;
        accCorrQuat[2] = 0;
        accCorrQuat[3] = 0;
    }
    quatMultiply(accCorrQuat, state.accQuat, state.accQuat);
    q30_normalize(state.accQuat, 4);

    // calculate correction angular rate to facilitate debugging
    // acos can be replaced by 2*arctan( sqrt(1-x*x) / x )
        state.lastAccCorrAngularRate = _IQ24toF(_IQ24acos((vqf_q24_t)accEarthFixed[2] * 256)) / coeffs.accTs;

    // bias estimation
    if (params.motionBiasEstEnabled || params.restBiasEstEnabled) {
        vqf_real_t biasClip = params.biasClip*(vqf_real_t)(M_PIf/180.0);

        vqf_q30_t accGyrQuat[4];
        vqf_q30_t rotation[9];
        vqf_real_t R[9];
        vqf_real_t biasLp[2];

        // get rotation matrix corresponding to accGyrQuat
        quatMultiply(state.accQuat, state.gyrQuat, accGyrQuat);
        quatToMatrix(accGyrQuat, rotation);
        for (size_t i = 0; i < 9; i++) {
            R[i] = q30_to_real(rotation[i]);
        }

        // calculate R*b_hat (only the x and y component, as z is not needed)
        biasLp[0] = R[0]*state.bias[0] + R[1]*state.bias[1] + R[2]*state.bias[2];
        biasLp[1] = R[3]*state.bias[0] + R[4]*state.bias[1] + R[5]*state.bias[2];

        // low-pass filter R and R*b_hat
        filterVec(R, 9, params.tauAcc, coeffs.accTs, coeffs.accLpB, coeffs.accLpA, state.motionBiasEstRLpState, R);
        filterVec(biasLp, 2, params.tauAcc, coeffs.accTs, coeffs.accLpB, coeffs.accLpA, state.motionBiasEstBiasLpState,
                  biasLp);

        // set measurement error and covariance for the respective Kalman filter update
        vqf_real_t w[3];
        vqf_real_t e[3];
        if (state.restDetected && params.restBiasEstEnabled) {
            e[0] = state.restLastGyrLp[0] - state.bias[0];
            e[1] = state.restLastGyrLp[1] - state.bias[1];
            e[2] = state.restLastGyrLp[2] - state.bias[2];
            matrix3SetToScaledIdentity(1.0, R);
            vqf_fill_real(w, 3, coeffs.biasRestW);
            // std::fill(w, w+3, coeffs.biasRestW);
        } else if (params.motionBiasEstEnabled) {
            e[0] = -accEarth[1]/coeffs.accTs + biasLp[0] - R[0]*state.bias[0] - R[1]*state.bias[1] - R[2]*state.bias[2];
            e[1] = accEarth[0]/coeffs.accTs + biasLp[1] - R[3]*state.bias[0] - R[4]*state.bias[1] - R[5]*state.bias[2];
            e[2] = - R[6]*state.bias[0] - R[7]*state.bias[1] - R[8]*state.bias[2];
            w[0] = coeffs.biasMotionW;
            w[1] = coeffs.biasMotionW;
            w[2] = coeffs.biasVerticalW;
        } else {
            vqf_fill_real(w, 3, -1);
            // std::fill(w, w+3, -1); // disable update
        }

        // Kalman filter update
        // step 1: P = P + V (also increase covariance if there is no measurement update!)
        if (state.biasP[0] < coeffs.biasP0) {
            state.biasP[0] += coeffs.biasV;
        }
        if (state.biasP[4] < coeffs.biasP0) {
            state.biasP[4] += coeffs.biasV;
        }
        if (state.biasP[8] < coeffs.biasP0) {
            state.biasP[8] += coeffs.biasV;
        }
        if (w[0] >= 0) {
            // clip disagreement to -2..2 °/s
            // (this also effectively limits the harm done by the first inclination correction step)
            clip(e, 3, -biasClip, biasClip);

            // step 2: K = P R^T inv(W + R P R^T)
            vqf_real_t K[9];
            matrix3MultiplyTpsSecond(state.biasP, R, K); // K = P R^T
            matrix3Multiply(R, K, K); // K = R P R^T
            K[0] += w[0];
            K[4] += w[1];
            K[8] += w[2]; // K = W + R P R^T
            matrix3Inv(K, K); // K = inv(W + R P R^T)
            matrix3MultiplyTpsFirst(R, K, K); // K = R^T inv(W + R P R^T)
            matrix3Multiply(state.biasP, K, K); // K = P R^T inv(W + R P R^T)

            // step 3: bias = bias + K (y - R bias) = bias + K e
            state.bias[0] += K[0]*e[0] + K[1]*e[1] + K[2]*e[2];
            state.bias[1] += K[3]*e[0] + K[4]*e[1] + K[5]*e[2];
            state.bias[2] += K[6]*e[0] + K[7]*e[1] + K[8]*e[2];

            // step 4: P = P - K R P
            matrix3Multiply(K, R, K); // K = K R
            matrix3Multiply(K, state.biasP, K); // K = K R P
            for(size_t i = 0; i < 9; i++) {
                state.biasP[i] -= K[i];
            }

            // clip bias estimate to -2..2 °/s
            clip(state.bias, 3, -biasClip, biasClip);
            for (size_t i = 0; i < 3; i++) {
                state.biasQ16[i] = q16_from_real(state.bias[i]);
            }
        }
    }
}

void updateMag(const vqf_real_t mag[3])
{
    // ignore [0 0 0] samples
    if (mag[0] == (vqf_real_t)(0.0) && mag[1] == (vqf_real_t)(0.0) && mag[2] == (vqf_real_t)(0.0)) {
        return;
    }

    vqf_q16_t magFixed[3];
    vqf_q16_t magEarthFixed[3];

    // bring magnetometer measurement into 6D earth frame
    vqf_q30_t accGyrQuat[4];
    quatMultiply(state.accQuat, state.gyrQuat, accGyrQuat);
    for (size_t i = 0; i < 3; i++) {
        magFixed[i] = q16_from_real(mag[i]);
    }
    quatRotate(accGyrQuat, magFixed, magEarthFixed);
    if (params.magDistRejectionEnabled) {
        vqf_q16_t magNorm = q16_norm(magEarthFixed, 3);
        state.magNormDip[0] = q16_to_real(magNorm);
        // asin can be replace by 2*arctan(x / sqrt(1-x*x))
        if (magNorm > 0) {
            vqf_q24_t ratio = q24_clamp_unit(
                    (vqf_q24_t)q16_div(magEarthFixed[2], magNorm) * 256);
            state.magNormDip[1] = -_IQ24toF(_IQ24asin(ratio));
        } else {
            state.magNormDip[1] = 0;
        }

        if (params.magCurrentTau > 0) {
            filterVec(state.magNormDip, 2, params.magCurrentTau, coeffs.magTs, coeffs.magNormDipLpB,
                      coeffs.magNormDipLpA, state.magNormDipLpState, state.magNormDip);
        }

        // magnetic disturbance detection
        // fabs_fast can be replaced by arm_abs_f32 from CMSIS-DSP
        if (fabs_fast(state.magNormDip[0] - state.magRefNorm) < params.magNormTh*state.magRefNorm
                && fabs_fast(state.magNormDip[1] - state.magRefDip) < params.magDipTh*(vqf_real_t)(M_PIf/180.0)) {
            state.magUndisturbedT += coeffs.magTs;
            if (state.magUndisturbedT >= params.magMinUndisturbedTime) {
                state.magDistDetected = false;
                state.magRefNorm += coeffs.kMagRef*(state.magNormDip[0] - state.magRefNorm);
                state.magRefDip += coeffs.kMagRef*(state.magNormDip[1] - state.magRefDip);
            }
        } else {
            state.magUndisturbedT = 0.0;
            state.magDistDetected = true;
        }

        // new magnetic field acceptance
        // fabs_fast can be replaced by arm_abs_f32 from CMSIS-DSP
        if (fabs_fast(state.magNormDip[0] - state.magCandidateNorm) < params.magNormTh*state.magCandidateNorm
                && fabs_fast(state.magNormDip[1] - state.magCandidateDip) < params.magDipTh*(vqf_real_t)(M_PIf/180.0)) {
            if (norm(state.restLastGyrLp, 3) >= params.magNewMinGyr*M_PIf/180.0) {
                state.magCandidateT += coeffs.magTs;
            }
            state.magCandidateNorm += coeffs.kMagRef*(state.magNormDip[0] - state.magCandidateNorm);
            state.magCandidateDip += coeffs.kMagRef*(state.magNormDip[1] - state.magCandidateDip);

            if (state.magDistDetected && (state.magCandidateT >= params.magNewTime || (
                    state.magRefNorm == 0.0 && state.magCandidateT >= params.magNewFirstTime))) {
                state.magRefNorm = state.magCandidateNorm;
                state.magRefDip = state.magCandidateDip;
                state.magDistDetected = false;
                state.magUndisturbedT = params.magMinUndisturbedTime;
            }
        } else {
            state.magCandidateT = 0.0;
            state.magCandidateNorm = state.magNormDip[0];
            state.magCandidateDip = state.magNormDip[1];
        }
    }

    // calculate disagreement angle based on current magnetometer measurement
    vqf_q24_t magX = (vqf_q24_t)magEarthFixed[0] * 256;
    vqf_q24_t magY = (vqf_q24_t)magEarthFixed[1] * 256;
    state.lastMagDisAngle = _IQ24toF(_IQ24atan2(magX, magY)) - state.delta;

    // make sure the disagreement angle is in the range [-pi, pi]
    if (state.lastMagDisAngle > (vqf_real_t)(M_PIf)) {
        state.lastMagDisAngle -= (vqf_real_t)(2*M_PIf);
    } else if (state.lastMagDisAngle < (vqf_real_t)(-M_PIf)) {
        state.lastMagDisAngle += (vqf_real_t)(2*M_PIf);
    }

    vqf_real_t k = coeffs.kMag;

    if (params.magDistRejectionEnabled) {
        // magnetic disturbance rejection
        if (state.magDistDetected) {
            if (state.magRejectT <= params.magMaxRejectionTime) {
                state.magRejectT += coeffs.magTs;
                k = 0;
            } else {
                k /= params.magRejectionFactor;
            }
        } else {
            state.magRejectT = vqf_max(state.magRejectT - params.magRejectionFactor*coeffs.magTs, (vqf_real_t)(0.0));
        }
    }

    // ensure fast initial convergence
    if (state.kMagInit != (vqf_real_t)(0.0)) {
        // make sure that the gain k is at least 1/N, N=1,2,3,... in the first few samples
        if (k < state.kMagInit) {
            k = state.kMagInit;
        }

        // iterative expression to calculate 1/N
        state.kMagInit = state.kMagInit/(state.kMagInit+1);

        // disable if t > tauMag
        if (state.kMagInit*params.tauMag < coeffs.magTs) {
            state.kMagInit = 0.0;
        }
    }

    // first-order filter step
    state.delta += k*state.lastMagDisAngle;
    // calculate correction angular rate to facilitate debugging
    state.lastMagCorrAngularRate = k*state.lastMagDisAngle/coeffs.magTs;

    // make sure delta is in the range [-pi, pi]
    if (state.delta > (vqf_real_t)(M_PIf)) {
        state.delta -= (vqf_real_t)(2*M_PIf);
    } else if (state.delta < (vqf_real_t)(-M_PIf)) {
        state.delta += (vqf_real_t)(2*M_PIf);
    }
}

//void update(const vqf_real_t gyr[3], const vqf_real_t acc[3])
//{
//    updateGyr(gyr);
//    updateAcc(acc);
//}
//
//void update(const vqf_real_t gyr[3], const vqf_real_t acc[3], const vqf_real_t mag[3])
//{
//    updateGyr(gyr);
//    updateAcc(acc);
//    updateMag(mag);
//}

// void updateBatch(const vqf_real_t gyr[], const vqf_real_t acc[], const vqf_real_t mag[], size_t N,
//                       vqf_real_t out6D[], vqf_real_t out9D[], vqf_real_t outDelta[], vqf_real_t outBias[],
//                       vqf_real_t outBiasSigma[], bool outRest[], bool outMagDist[])
// {
//     for (size_t i = 0; i < N; i++) {
//         if (mag) {
//             update(gyr+3*i, acc+3*i, mag+3*i);
//         } else {
//             update(gyr+3*i, acc+3*i);
//         }
//         if (out6D) {
//             getQuat6D(out6D+4*i);
//         }
//         if (out9D) {
//             getQuat9D(out9D+4*i);
//         }
//         if (outDelta) {
//             outDelta[i] = state.delta;
//         }
//         if (outBias) {
//             memcpy(outBias+3*i, state.bias, sizeof(state.bias));
//             // std::copy(state.bias, state.bias+3, outBias+3*i);
//         }
//         if (outBiasSigma) {
//             outBiasSigma[i] = getBiasEstimate({0,0,0});
//         }
//         if (outRest) {
//             outRest[i] = state.restDetected;
//         }
//         if (outMagDist) {
//             outMagDist[i] = state.magDistDetected;
//         }
//     }
// }

void getQuat3D(vqf_real_t out[4])
{
    for (size_t i = 0; i < 4; i++) {
        out[i] = q30_to_real(state.gyrQuat[i]);
    }
}

void getQuat6D(vqf_real_t out[4])
{
    vqf_q30_t quat[4];
    quatMultiply(state.accQuat, state.gyrQuat, quat);
    for (size_t i = 0; i < 4; i++) {
        out[i] = q30_to_real(quat[i]);
    }
}

void getQuat9D(vqf_real_t out[4])
{
    vqf_q30_t quat[4];
    quatMultiply(state.accQuat, state.gyrQuat, quat);
    vqf_q30_t corrected[4];
    quatApplyDelta(quat, state.delta, corrected);
    for (size_t i = 0; i < 4; i++) {
        out[i] = q30_to_real(corrected[i]);
    }
}

vqf_real_t getDelta()
{
    return state.delta;
}

vqf_real_t getBiasEstimate(vqf_real_t out[3])
{
    if (out) {
        memcpy(out, state.bias, sizeof(state.bias));
        // std::copy(state.bias, state.bias+3, out);
    }
    // use largest absolute row sum as upper bound estimate for largest eigenvalue (Gershgorin circle theorem)
    // and clip output to biasSigmaInit
    // fabs_fast can be replaced by arm_abs_f32 from CMSIS-DSP
    vqf_real_t sum1 = fabs_fast(state.biasP[0]) + fabs_fast(state.biasP[1]) + fabs_fast(state.biasP[2]);
    vqf_real_t sum2 = fabs_fast(state.biasP[3]) + fabs_fast(state.biasP[4]) + fabs_fast(state.biasP[5]);
    vqf_real_t sum3 = fabs_fast(state.biasP[6]) + fabs_fast(state.biasP[7]) + fabs_fast(state.biasP[8]);
    vqf_real_t P = vqf_min(vqf_max(vqf_max(sum1, sum2), sum3), coeffs.biasP0);
    // convert standard deviation from 0.01deg to rad
    // sqrt can be replaced by arm_sqrt_f32 from CMSIS_DSP
    return sqrt_fast(P)*(vqf_real_t)(M_PIf/100.0/180.0);
}

void setBiasEstimate(vqf_real_t bias[3], vqf_real_t sigma)
{
    memcpy(state.bias, bias, sizeof(vqf_real_t[3]));
    for (size_t i = 0; i < 3; i++) {
        state.biasQ16[i] = q16_from_real(bias[i]);
    }
    // std::copy(bias, bias+3, state.bias);
    if (sigma > 0) {
        vqf_real_t P = vqf_square(sigma*(vqf_real_t)(180.0*100.0/M_PIf));
        matrix3SetToScaledIdentity(P, state.biasP);
    }
}

bool getRestDetected()
{
    return state.restDetected;
}

bool getMagDistDetected()
{
    return state.magDistDetected;
}

void getRelativeRestDeviations(vqf_real_t out[2])
{
    // sqrt can be replaced by arm_sqrt_f32 from CMSIS_DSP
    out[0] = sqrt_fast(state.restLastSquaredDeviations[0]) / (params.restThGyr*(vqf_real_t)(M_PIf/180.0));
    out[1] = sqrt_fast(state.restLastSquaredDeviations[1]) / params.restThAcc;
}

vqf_real_t getMagRefNorm()
{
    return state.magRefNorm;
}

vqf_real_t getMagRefDip()
{
    return state.magRefDip;
}

void setMagRef(vqf_real_t norm, vqf_real_t dip)
{
    state.magRefNorm = norm;
    state.magRefDip = dip;
}

void setMotionBiasEstEnabled(bool enabled)
{
    if (params.motionBiasEstEnabled == enabled) {
        return;
    }
    params.motionBiasEstEnabled = enabled;
    vqf_fill_double(state.motionBiasEstRLpState, 9*2, NaN);
    // std::fill(state.motionBiasEstRLpState, state.motionBiasEstRLpState + 9*2, NaN);
    vqf_fill_double(state.motionBiasEstBiasLpState, 2*2, NaN);
    // std::fill(state.motionBiasEstBiasLpState, state.motionBiasEstBiasLpState + 2*2, NaN);
}

void setRestBiasEstEnabled(bool enabled)
{
    if (params.restBiasEstEnabled == enabled) {
        return;
    }
    params.restBiasEstEnabled = enabled;
    state.restDetected = false;

    vqf_fill_real(state.restLastSquaredDeviations, 2, 0.0);
    // std::fill(state.restLastSquaredDeviations, state.restLastSquaredDeviations + 3, 0.0);
    state.restT = 0.0;
    vqf_fill_real(state.restLastGyrLp, 3, 0.0);
    // std::fill(state.restLastGyrLp, state.restLastGyrLp + 3, 0.0);
    vqf_fill_double(state.restGyrLpState, 3*2, NaN);
    // std::fill(state.restGyrLpState, state.restGyrLpState + 3*2, NaN);
    vqf_fill_real(state.restLastAccLp, 3, 0.0);
    // std::fill(state.restLastAccLp, state.restLastAccLp + 3, 0.0);
    vqf_fill_double(state.restAccLpState, 3*2, NaN);
    // std::fill(state.restAccLpState, state.restAccLpState + 3*2, NaN);
}

void setMagDistRejectionEnabled(bool enabled)
{
    if (params.magDistRejectionEnabled == enabled) {
        return;
    }
    params.magDistRejectionEnabled = enabled;
    state.magDistDetected = true;
    state.magRefNorm = 0.0;
    state.magRefDip = 0.0;
    state.magUndisturbedT = 0.0;
    state.magRejectT = params.magMaxRejectionTime;
    state.magCandidateNorm = -1.0;
    state.magCandidateDip = 0.0;
    state.magCandidateT = 0.0;
    vqf_fill_double(state.magNormDipLpState, 2*2, NaN);
    // std::fill(state.magNormDipLpState, state.magNormDipLpState + 2*2, NaN);
}

void setTauAcc(vqf_real_t tauAcc)
{
    if (params.tauAcc == tauAcc) {
        return;
    }
    params.tauAcc = tauAcc;
    vqf_double_t newB[3];
    vqf_double_t newA[2];

    filterCoeffs(params.tauAcc, coeffs.accTs, newB, newA);
    filterAdaptStateForCoeffChange(state.lastAccLp, 3, coeffs.accLpB, coeffs.accLpA, newB, newA, state.accLpState);

    // For R and biasLP, the last value is not saved in the state.
    // Since b0 is small (at reasonable settings), the last output is close to state[0].
    vqf_real_t R[9];
    for (size_t i = 0; i < 9; i++) {
        R[i] = (vqf_real_t)state.motionBiasEstRLpState[2*i];
    }
    filterAdaptStateForCoeffChange(R, 9, coeffs.accLpB, coeffs.accLpA, newB, newA, state.motionBiasEstRLpState);
    vqf_real_t biasLp[2];
    for (size_t i = 0; i < 2; i++) {
        biasLp[i] = (vqf_real_t)state.motionBiasEstBiasLpState[2*i];
    }
    filterAdaptStateForCoeffChange(biasLp, 2, coeffs.accLpB, coeffs.accLpA, newB, newA, state.motionBiasEstBiasLpState);

    memcpy(coeffs.accLpB, newB, sizeof(newB));
    // std::copy(newB, newB+3, coeffs.accLpB);
    memcpy(coeffs.accLpA, newA, sizeof(newA));
    // std::copy(newA, newA+2, coeffs.accLpA);
}

void setTauMag(vqf_real_t tauMag)
{
    params.tauMag = tauMag;
    coeffs.kMag = gainFromTau(params.tauMag, coeffs.magTs);
}

void setRestDetectionThresholds(vqf_real_t thGyr, vqf_real_t thAcc)
{
    params.restThGyr = thGyr;
    params.restThAcc = thAcc;
}

void resetState()
{
    quatSetToIdentity(state.gyrQuat);
    quatSetToIdentity(state.accQuat);
    state.delta = 0.0;

    state.restDetected = false;
    state.magDistDetected = true;

    vqf_fill_real(state.lastAccLp, 3, 0);
    // std::fill(state.lastAccLp, state.lastAccLp+3, 0);
    vqf_fill_double(state.accLpState, 3*2, NaN);
    // std::fill(state.accLpState, state.accLpState + 3*2, NaN);
    state.lastAccCorrAngularRate = 0.0;

    state.kMagInit = 1.0;
    state.lastMagDisAngle = 0.0;
    state.lastMagCorrAngularRate = 0.0;

    vqf_fill_real(state.bias, 3, 0);
    // std::fill(state.bias, state.bias+3, 0);
    state.biasQ16[0] = 0;
    state.biasQ16[1] = 0;
    state.biasQ16[2] = 0;

    matrix3SetToScaledIdentity(coeffs.biasP0, state.biasP);

    vqf_fill_double(state.motionBiasEstRLpState, 9*2, NaN);
    // std::fill(state.motionBiasEstRLpState, state.motionBiasEstRLpState + 9*2, NaN);
    vqf_fill_double(state.motionBiasEstBiasLpState, 2*2, NaN);
    // std::fill(state.motionBiasEstBiasLpState, state.motionBiasEstBiasLpState + 2*2, NaN);


    vqf_fill_real(state.restLastSquaredDeviations, 2, 0.0);
    // std::fill(state.restLastSquaredDeviations, state.restLastSquaredDeviations + 3, 0.0);
    state.restT = 0.0;
    vqf_fill_real(state.restLastGyrLp, 3, 0.0);
    // std::fill(state.restLastGyrLp, state.restLastGyrLp + 3, 0.0);
    vqf_fill_double(state.restGyrLpState, 3*2, NaN);
    // std::fill(state.restGyrLpState, state.restGyrLpState + 3*2, NaN);
    vqf_fill_real(state.restLastAccLp, 3, 0.0);
    // std::fill(state.restLastAccLp, state.restLastAccLp + 3, 0.0);
    vqf_fill_double(state.restAccLpState, 3*2, NaN);
    // std::fill(state.restAccLpState, state.restAccLpState + 3*2, NaN);

    state.magRefNorm = 0.0;
    state.magRefDip = 0.0;
    state.magUndisturbedT = 0.0;
    state.magRejectT = params.magMaxRejectionTime;
    state.magCandidateNorm = -1.0;
    state.magCandidateDip = 0.0;
    state.magCandidateT = 0.0;
    vqf_fill_real(state.magNormDip, 2, 0);
    // std::fill(state.magNormDip, state.magNormDip + 2, 0);
    vqf_fill_double(state.magNormDipLpState, 2*2, NaN);
    // std::fill(state.magNormDipLpState, state.magNormDipLpState + 2*2, NaN);
}

void setup()
{
    // assert(coeffs.gyrTs > 0);
    // assert(coeffs.accTs > 0);
    // assert(coeffs.magTs > 0);

    filterCoeffs(params.tauAcc, coeffs.accTs, coeffs.accLpB, coeffs.accLpA);

    coeffs.kMag = gainFromTau(params.tauMag, coeffs.magTs);

    coeffs.biasP0 = vqf_square(params.biasSigmaInit*100.0f);
    // the system noise increases the variance from 0 to (0.1 °/s)^2 in biasForgettingTime seconds
    coeffs.biasV = vqf_square(0.1f*100.0f)*coeffs.accTs/params.biasForgettingTime;


    vqf_real_t pMotion = vqf_square(params.biasSigmaMotion*100.0f);
    coeffs.biasMotionW = vqf_square(pMotion) / coeffs.biasV + pMotion;
    coeffs.biasVerticalW = coeffs.biasMotionW / vqf_max(params.biasVerticalForgettingFactor, (vqf_real_t)(1e-10));


    vqf_real_t pRest = vqf_square(params.biasSigmaRest*100.0f);
    coeffs.biasRestW = vqf_square(pRest) / coeffs.biasV + pRest;

    filterCoeffs(params.restFilterTau, coeffs.gyrTs, coeffs.restGyrLpB, coeffs.restGyrLpA);
    filterCoeffs(params.restFilterTau, coeffs.accTs, coeffs.restAccLpB, coeffs.restAccLpA);

    coeffs.kMagRef = gainFromTau(params.magRefTau, coeffs.magTs);
    if (params.magCurrentTau > 0) {
        filterCoeffs(params.magCurrentTau, coeffs.magTs, coeffs.magNormDipLpB, coeffs.magNormDipLpA);
    } else {
        vqf_fill_double(coeffs.magNormDipLpB, 3, NaN);
        // std::fill(coeffs.magNormDipLpB, coeffs.magNormDipLpB + 3, NaN);
        vqf_fill_double(coeffs.magNormDipLpA, 2, NaN);
        // std::fill(coeffs.magNormDipLpA, coeffs.magNormDipLpA + 2, NaN);
    }

    resetState();
}


void initVqf(vqf_real_t gyrTs, vqf_real_t accTs, vqf_real_t magTs)
{
    coeffs.gyrTs = gyrTs;
    coeffs.accTs = accTs > 0 ? accTs : gyrTs;
    coeffs.magTs = magTs > 0 ? magTs : gyrTs;

    init_params();
    setup();
}
