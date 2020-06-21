const size_t NTB = 256;
const size_t EXT = 8;
#define divCeil(a, b) (((a) + (b) - 1) / (b))
struct Ctx {
    float *x, *y, *r;
    size_t n;
};

__global__ void devDot(float *x, float *y, size_t n, float *r) {
    __shared__ float rb[NTB];
    size_t itb = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x * EXT + itb;
    float s = 0.0;
    for (size_t j = 0; j < EXT && i < n; j++, i += blockDim.x) {
        s += x[i] * y[i];
    }

    rb[itb] = s;
    __syncthreads();
    for (size_t i = NTB >> 1; i != 0; i >>= 1) {
        if (itb < i) rb[itb] += rb[itb + i];
        __syncthreads();
    }
    if (0 == itb) r[blockIdx.x] = rb[0];
}

extern "C" __declspec(dllexport) void getInputs(Ctx *ctx, float **px, float **py) {
    *px = ctx->x;
    *py = ctx->y;
}

extern "C" __declspec(dllexport) void init(Ctx **p, size_t n) {
    Ctx *ctx = (Ctx *)malloc(sizeof(Ctx));
    ctx->n = n;
    size_t sz = sizeof(float) * n;
    cudaMallocManaged(&(ctx->x), sz);
    cudaMallocManaged(&(ctx->y), sz);
    cudaMallocManaged(&(ctx->r), sizeof(float) * divCeil(n, NTB) / EXT);
    *p = ctx;
}

extern "C" __declspec(dllexport) void deinit(Ctx *ctx) {
    cudaFree(ctx->x);
    cudaFree(ctx->y);
    cudaFree(ctx->r);
    free(ctx);
}
extern "C" __declspec(dllexport) void dot(Ctx *ctx, float *r) {
    size_t nb = divCeil(ctx->n, NTB) / EXT;
    float *rd = ctx->r;
    devDot<<<nb, NTB>>>(ctx->x, ctx->y, ctx->n, rd);
    cudaDeviceSynchronize();
    float s = 0.0;
    for (size_t i = 0; i < nb; i++) s += rd[i];
    *r = s;
}