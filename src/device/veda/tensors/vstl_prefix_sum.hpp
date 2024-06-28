#pragma once

#define PREFIX_SUM_NAME		prefix_sum_no
#define PREFIX_SUM_VLEN		256
#define PREFIX_SUM_VREG		""
#define PREFIX_SUM_VECTOR	"_NEC vector"
#include "vstl_prefix_sum.inc.hpp"
#undef PREFIX_SUM_NAME
#undef PREFIX_SUM_VLEN
#undef PREFIX_SUM_VREG
#undef PREFIX_SUM_VECTOR

#define PREFIX_SUM_NAME		prefix_sum_vec
#define PREFIX_SUM_VLEN		256
#define PREFIX_SUM_VREG		"_NEC vreg(current_val)"
#define PREFIX_SUM_VECTOR	"_NEC vector"
#include "vstl_prefix_sum.inc.hpp"
#undef PREFIX_SUM_NAME
#undef PREFIX_SUM_VLEN
#undef PREFIX_SUM_VREG
#undef PREFIX_SUM_VECTOR

#define PREFIX_SUM_NAME		prefix_sum_packed
#define PREFIX_SUM_VLEN		512
#define PREFIX_SUM_VREG		"_NEC pvreg(current_val)"
#define PREFIX_SUM_VECTOR	"_NEC packed_vector"
#include "vstl_prefix_sum.inc.hpp"
#undef PREFIX_SUM_NAME
#undef PREFIX_SUM_VLEN
#undef PREFIX_SUM_VREG
#undef PREFIX_SUM_VECTOR

namespace vstl {
	namespace seq {
//------------------------------------------------------------------------------
template<typename T>	void	prefix_sum			(const T* valp, T* outp, size_t size)				{	prefix_sum_no		(valp, outp, size);	}
template<>				void	prefix_sum<float>	(const float* valp, float* outp, size_t size)		{	prefix_sum_packed	(valp, outp, size);	}
template<>				void	prefix_sum<double>	(const double* valp, double* outp, size_t size)		{	prefix_sum_vec		(valp, outp, size);	}
template<>				void	prefix_sum<int32_t>	(const int32_t* valp, int32_t* outp, size_t size)	{	prefix_sum_packed	(valp, outp, size);	}
template<>				void	prefix_sum<uint32_t>(const uint32_t* valp, uint32_t* outp, size_t size)	{	prefix_sum_packed	(valp, outp, size);	}

//------------------------------------------------------------------------------
	}
}
