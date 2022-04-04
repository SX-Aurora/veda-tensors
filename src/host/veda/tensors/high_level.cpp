#include "internal.h"

//------------------------------------------------------------------------------
#define VERIFY(...)	try { __VA_ARGS__ } catch(VEDAresult res) { return res;	}

//------------------------------------------------------------------------------
inline void verify_axis(const VEDATensors_tensor* t, const int axis) {
	VEDA_TENSORS_THROW(axis < 0 || axis >= t->dims, "Expected axis to be in range 0 and %i but is %i.", t->dims-1, axis);
}

//------------------------------------------------------------------------------
inline void verify_ptr(const void* ptr) {
	VEDA_TENSORS_THROW(ptr == 0, "Missing host pointer!");
}

//------------------------------------------------------------------------------
inline std::tuple<size_t, size_t, size_t> compute_lcr(const VEDATensors_tensor* t, const int axis) {
	size_t left = 1, center = t->shape[axis], right = 1;
	for(int i = 0;        i < axis;    i++)	left  *= t->shape[i];
	for(int i = axis + 1; i < t->dims; i++)	right *= t->shape[i];
	return {left, center, right};
}

//------------------------------------------------------------------------------
// verify_same_shapes
//------------------------------------------------------------------------------
inline void verify_same_shapes(VEDATensors_tensor* ref, VEDATensors_tensor* t) {
	VEDA_TENSORS_THROW(ref->dims != t->dims, "Expected %i dims, but found %i.", ref->dims, t->dims);
	VEDA_TENSORS_THROW(ref->numel != t->numel, "Expected %llu numels, but found %llu.", ref->numel, t->numel);
#ifndef NDEBUG
	for(int i = 0; i < ref->dims; i++)
		VEDA_TENSORS_THROW(ref->shape[i] != t->shape[i], "Expected size %llu, but found %llu in dim %i.", ref->shape[i], t->shape[i], i);
#endif
}

//------------------------------------------------------------------------------
template<typename... Args>
inline void verify_same_shapes(VEDATensors_tensor* a, VEDATensors_tensor* b, Args... vargs) {
	verify_same_shapes(a, b);
	verify_same_shapes(a, vargs...);
}

//------------------------------------------------------------------------------
// verify_same_dtype
//------------------------------------------------------------------------------
inline void verify_same_dtype(const VEDATensors_dtype ref, VEDATensors_tensor* t) {
	VEDA_TENSORS_THROW(ref != t->dtype, "Excepted %s but found %s.", veda_tensors_get_dtype(ref), veda_tensors_get_dtype(t->dtype));
}

//------------------------------------------------------------------------------
template<typename... Args>
inline void verify_same_dtype(const VEDATensors_dtype ref, VEDATensors_tensor* t, Args... vargs) {
	verify_same_dtype(ref, t);
	verify_same_dtype(ref, vargs...);
}

//------------------------------------------------------------------------------
template<typename... Args>
inline void verify_same_dtype(VEDATensors_tensor* t, Args... vargs) {
	verify_same_dtype(t->dtype, vargs...);
}

//------------------------------------------------------------------------------
// verify_tensors
//------------------------------------------------------------------------------
inline void verify_tensors(VEDATensors_tensor* t) {
	VEDA_TENSORS_THROW(t == 0, "Missing tensor!");
	VEDA_TENSORS_THROW(t->dims <= 0 || t->dims > VEDA_TENSORS_MAX_DIMS, "Expected dims to be between 1 and %i but are %i.", VEDA_TENSORS_MAX_DIMS, t->dims);
#ifndef NDEBUG
	size_t numel = 1;
	for(int i = 0; i < t->dims; i++)
		numel *= t->shape[i];
	VEDA_TENSORS_THROW(numel != t->numel, "Expected numel to be %llu but are %llu.", numel, t->numel);
#endif
}

//------------------------------------------------------------------------------
template<typename... Args>
inline void verify_tensors(VEDATensors_tensor* t, Args... vargs) {
	verify_tensors(t);
	verify_tensors(vargs...);
}

//------------------------------------------------------------------------------
// API
//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_binary(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x, VEDATensors_tensor* y, const VEDATensors_binary_op op) {
	VERIFY(
		verify_tensors		(o, x, y);
		verify_same_dtype	(x, y);
		verify_same_dtype	(VEDA_TENSORS_DTYPE_S8, o);
	)
	return veda_tensors_ll_binary(handle, o->ptr, x->ptr, y->ptr, o->numel, x->numel, y->numel, op, x->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_bitwise(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x, VEDATensors_tensor* y, const VEDATensors_bitwise_op op) {
	VERIFY(
		verify_tensors		(o, x, y);
		verify_same_dtype	(o, x, y);
	)
	return veda_tensors_ll_bitwise(handle, o->ptr, x->ptr, y->ptr, o->numel, x->numel, y->numel, op, x->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_cat(VEDATensors_chandle handle, const int inputCnt, VEDATensors_tensor* inputs, VEDATensors_tensor* output, const int dim) {
	VERIFY(
		verify_tensors(output);
		auto dtype = output->dtype;
		for(int i = 0; i < inputCnt; i++) {
			auto& in = inputs[i];
			verify_tensors		(&in);
			verify_same_dtype	(dtype, &in);
		}
	)

	std::vector<VEDAdeviceptr> ptrs;	ptrs	.reserve(inputCnt);
	std::vector<size_t> dimSizes;		dimSizes.reserve(inputCnt);

	for(int i = 0; i < inputCnt; i++) {
		auto& in = inputs[i];
		ptrs	.emplace_back(in.ptr);
		dimSizes.emplace_back(in.shape[dim]);
	}

	return veda_tensors_ll_cat(handle, inputCnt, ptrs.data(), dimSizes.data(), output->dims, output->shape, output->ptr, dim, output->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_convert(VEDATensors_chandle handle, VEDATensors_tensor* dst, VEDATensors_tensor* src) {
	VERIFY(
		verify_tensors		(dst, src);
		verify_same_shapes	(dst, src);
	)
	return veda_tensors_ll_convert(handle, dst->ptr, src->ptr, dst->numel, dst->dtype, src->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_copy(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x) {
	VERIFY(
		verify_tensors		(o, x);
		verify_same_dtype	(o, x);
	)
	return veda_tensors_ll_copy(handle, o->ptr, x->ptr, o->numel, x->numel, o->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_count(VEDATensors_chandle handle, VEDATensors_tensor* in, size_t* cnt) {
	VERIFY(
		verify_tensors	(in);
		verify_ptr		(cnt);
	)
	return veda_tensors_ll_count(handle, in->ptr, in->numel, in->dtype, cnt);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_fill(VEDATensors_chandle handle, VEDATensors_tensor* in, const VEDATensors_scalar value) {
	VERIFY(verify_tensors(in);)
	return veda_tensors_ll_fill(handle, in->ptr, value, in->numel, in->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_masked_select(VEDATensors_chandle handle, VEDATensors_tensor* out, VEDATensors_tensor* in, VEDATensors_tensor* mask) {
	VERIFY(
		verify_tensors		(out, in, mask);
		verify_same_shapes	(in, mask);
		verify_same_dtype	(out, in);
		verify_same_dtype	(VEDA_TENSORS_DTYPE_S8, mask);
	)
	return veda_tensors_ll_masked_select(handle, out->ptr, in->ptr, mask->ptr, out->numel, in->numel, out->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_masked_scatter(VEDATensors_chandle handle, VEDATensors_tensor* out, VEDATensors_tensor* in, VEDATensors_tensor* mask) {
	VERIFY(
		verify_tensors		(out, in, mask);
		verify_same_shapes	(out, in, mask);
		verify_same_dtype	(out, in);
		verify_same_dtype	(VEDA_TENSORS_DTYPE_S8, mask);
	)
	return veda_tensors_ll_masked_scatter(handle, out->ptr, in->ptr, mask->ptr, out->numel, out->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_masked_fill(VEDATensors_chandle handle, VEDATensors_tensor* out, const VEDATensors_scalar value, VEDATensors_tensor* mask) {
	VERIFY(
		verify_tensors		(out, mask);
		verify_same_shapes	(out, mask);
		verify_same_dtype	(VEDA_TENSORS_DTYPE_S8, mask);
	)
	return veda_tensors_ll_masked_fill(handle, out->ptr, value, mask->ptr, out->numel, out->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_masked_fill_t(VEDATensors_chandle handle, VEDATensors_tensor* out, VEDATensors_tensor* value, VEDATensors_tensor* mask) {
	VERIFY(
		verify_tensors			(out, value, mask);
		verify_same_shapes		(out, mask);
		if(value->numel > 1)
			verify_same_shapes	(out, value);
		verify_same_dtype		(out, value);
		verify_same_dtype		(VEDA_TENSORS_DTYPE_S8, mask);
	)
	return veda_tensors_ll_masked_fill_t(handle, out->ptr, value->ptr, mask->ptr, out->numel, value->numel, out->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_prefix_sum(VEDATensors_chandle handle, VEDATensors_tensor* out, VEDATensors_tensor* carry, VEDATensors_tensor* in, const int axis, const int inclusive) {
	VERIFY(
		verify_tensors		(out, in);
		verify_same_dtype	(out, in);
		verify_same_shapes	(out, in);
		verify_axis			(out, axis);
		if(carry) {
			verify_tensors		(carry);
			verify_same_dtype	(out, carry);
		}
	)

	auto [left, center, right] = compute_lcr(in, axis);
	return veda_tensors_ll_prefix_sum(handle, out->ptr, carry ? carry->ptr : 0, in->ptr, left, center, right, out->dtype, inclusive);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_print(VEDATensors_chandle handle, VEDATensors_tensor* in) {
	VERIFY(
		verify_tensors(in);
	)
	return veda_tensors_ll_print(handle, in->ptr, in->numel, in->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_reduce(VEDATensors_chandle handle, VEDATensors_tensor* out, VEDATensors_tensor* in, const VEDATensors_reduce_op op) {
	VERIFY(
		verify_tensors		(out, in);
		verify_same_dtype	(out, in);
	)
	return veda_tensors_ll_reduce(handle, out->ptr, in->ptr, in->numel, op, out->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_reduce_dim(VEDATensors_chandle handle, VEDATensors_tensor* values, VEDATensors_tensor* indices, VEDATensors_tensor* in, const VEDATensors_reduce_op op, const int axis) {
	VERIFY(
		verify_tensors		(values, in);
		verify_same_dtype	(values, in);
		verify_axis			(values, axis);
		if(indices) {
			verify_tensors		(indices);
			verify_same_dtype	(VEDA_TENSORS_DTYPE_S64, indices);
			verify_same_shapes	(values, indices);
		}
	)

	auto [left, center, right] = compute_lcr(in, axis);
	return veda_tensors_ll_reduce_dim(handle, values->ptr, indices ? indices->ptr : 0, in->ptr, left, center, right, op, in->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_reduce_scalar(VEDATensors_chandle handle, VEDATensors_tensor* in, const VEDATensors_reduce_op op, VEDATensors_scalar* result) {
	VERIFY(
		verify_tensors	(in);
		verify_ptr		(result);
	)
	return veda_tensors_ll_reduce_scalar(handle, in->ptr, in->numel, op, in->dtype, result);
}

//------------------------------------------------------------------------------
VEDAresult veda_tensors_select(VEDATensors_chandle handle, VEDATensors_tensor* out, VEDATensors_tensor* in, const size_t ocnt, const size_t ostride, const size_t icnt, const size_t offset) {
	VERIFY(
		verify_tensors		(out, in);
		verify_same_dtype	(out, in);
	)
	return veda_tensors_ll_select(handle, out->ptr, in->ptr, ocnt, ostride, icnt, offset, out->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_transpose(VEDATensors_chandle handle, VEDATensors_tensor* dst, VEDATensors_tensor* src) {
	VERIFY(
		verify_tensors		(dst, src);
		verify_same_dtype	(dst, src);
		THROWIF(dst->dims != 2, "Expected dst to be 2D but is %iD", dst->dims);
		THROWIF(src->dims != 2, "Expected dst to be 2D but is %iD", src->dims);
		THROWIF(dst->shape[0] != src->shape[1] || dst->shape[1] != src->shape[0], "Incompatible shapes found src: [%llu, %llu], dst: [%llu, %llu]", src->shape[0], src->shape[1], dst->shape[0], dst->shape[1]);
	)
	return veda_tensors_ll_transpose(handle, dst->ptr, src->ptr, dst->shape[0], dst->shape[1], dst->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_unary_b(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x, const VEDATensors_unary_op op) {
	VERIFY(
		verify_tensors		(o, x);
		verify_same_dtype	(VEDA_TENSORS_DTYPE_S8, o);
	)
	return veda_tensors_ll_unary_b(handle, o->ptr, x->ptr, o->numel, x->numel, op, x->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_unary_c(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x, const VEDATensors_unary_op op) {
	VERIFY(
		verify_tensors(o, x);
		if		(o->dtype == VEDA_TENSORS_DTYPE_F32_F32)	verify_same_dtype(VEDA_TENSORS_DTYPE_F32, x);
		else if	(o->dtype == VEDA_TENSORS_DTYPE_F64_F64)	verify_same_dtype(VEDA_TENSORS_DTYPE_F64, x);
	)
	return veda_tensors_ll_unary_c(handle, o->ptr, x->ptr, o->numel, x->numel, op, x->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_unary_t(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x, const VEDATensors_unary_op op) {
	VERIFY(
		verify_tensors		(o, x);
		verify_same_dtype	(o, x);
	)
	return veda_tensors_ll_unary_t(handle, o->ptr, x->ptr, o->numel, x->numel, op, x->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_unary_tt(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x, VEDATensors_tensor* y, const VEDATensors_unary_op op) {
	VERIFY(
		verify_tensors		(o, x, y);
		verify_same_dtype	(o, x, y);
	)
	return veda_tensors_ll_unary_tt(handle, o->ptr, x->ptr, y->ptr, o->numel, x->numel, y->numel, op, x->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_unary_tts(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x, VEDATensors_tensor* y, const VEDATensors_scalar alpha, const VEDATensors_unary_op op) {
	VERIFY(
		verify_tensors		(o, x, y);
		verify_same_dtype	(o, x, y);
	)
	return veda_tensors_ll_unary_tts(handle, o->ptr, x->ptr, y->ptr, alpha, o->numel, x->numel, y->numel, op, x->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_unary_ttts(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x, VEDATensors_tensor* y, VEDATensors_tensor* z, const VEDATensors_scalar alpha, const VEDATensors_unary_op op) {
	VERIFY(
		verify_tensors		(o, x, y, z);
		verify_same_dtype	(o, x, y, z);
	)
	return veda_tensors_ll_unary_ttts(handle, o->ptr, x->ptr, y->ptr, z->ptr, alpha, o->numel, x->numel, y->numel, z->numel, op, x->dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API void veda_tensors_debug(VEDATensors_tensor* t) {
	static_assert(VEDA_TENSORS_MAX_DIMS == 8); // needs to be updated if this changes
	ASSERT(t);
	L_INFO("VEDATensors_tensor [ptr: %p, dtype: %s, dims: %i, numel: %llu, shape: [%llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu]",
		t->ptr, veda_tensors_get_dtype(t->dtype), t->dims, t->numel, t->shape[0], t->shape[1], t->shape[2], t->shape[3], t->shape[4], t->shape[5], t->shape[6], t->shape[7]);
}

//------------------------------------------------------------------------------
