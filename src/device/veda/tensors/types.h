// Copyright 2022 NEC Laboratories Europe

#pragma once 

static_assert(sizeof(double)	== 8);
static_assert(sizeof(float)		== 4);
static_assert(sizeof(int)		== 4);
static_assert(sizeof(int16_t)	== 2);
static_assert(sizeof(int32_t)	== 4);
static_assert(sizeof(int64_t)	== 8);
static_assert(sizeof(int8_t)	== 1);
static_assert(sizeof(size_t)	== 8);
static_assert(sizeof(uint16_t)	== 2);
static_assert(sizeof(uint32_t)	== 4);
static_assert(sizeof(uint64_t)	== 8);
static_assert(sizeof(uint8_t)	== 1);

#define GEN(S, T)\
struct T { S x, y; };\
inline	T		operator+	(const T A, const T B)	{	T x; x.x = A.x + B.x;	x.y = A.y + B.y;	return x;	}\
inline	T		operator-	(const T A, const T B)	{	T x; x.x = A.x - B.x;	x.y = A.y - B.y;	return x;	}\
inline	T		operator+	(const T A, const S B)	{	T x; x.x = A.x + B;		x.y = A.y + B;		return x;	}\
inline	T		operator-	(const T A, const S B)	{	T x; x.x = A.x - B;		x.y = A.y - B;		return x;	}\
inline	T		operator*	(const T A, const S B)	{	T x; x.x = A.x * B;		x.y = A.y * B;		return x;	}\
inline	T		operator/	(const T A, const S B)	{	T x; x.x = A.x / B;		x.y = A.y / B;		return x;	}\
inline	T		operator+	(const S A, const T B)	{	T x; x.x = A + B.x;		x.y = A + B.y;		return x;	}\
inline	T		operator-	(const S A, const T B)	{	T x; x.x = A - B.x;		x.y = A - B.y;		return x;	}\
inline	T		operator*	(const S A, const T B)	{	T x; x.x = A * B.x;		x.y = A * B.y;		return x;	}\
inline	T		operator/	(const S A, const T B)	{	T x; x.x = A / B.x;		x.y = A / B.y;		return x;	}\
inline	bool	operator==	(const T A, const T B)	{	return A.x == B.x && A.y == B.y;						}\
inline	bool	operator!=	(const T A, const T B)	{	return A.x != B.x || A.y != B.y;						}\
inline	bool	operator==	(const S A, const T B)	{	return A   == B.x && A   == B.y;						}\
inline	bool	operator!=	(const S A, const T B)	{	return A   != B.x || A   != B.y;						}\
inline	bool	operator==	(const T A, const S B)	{	return A.x == B   && A.y == B;							}\
inline	bool	operator!=	(const T A, const S B)	{	return A.x != B   || A.y != B;							}\
inline	T		operator*	(const T A, const T B)	{\
	T x;\
	x.x = A.x * B.x - A.y * B.y;\
	x.y = A.y * B.x + A.x * B.y;\
	return x;\
}\
inline T operator/(const T A, const T B) {\
	auto d = (B.x*B.x + B.y*B.y);\
	T x;\
	x.x = (A.x*B.x + A.y*B.y)/d;\
	x.y = (A.y*B.x-A.x*B.y)/d;\
	return x;\
}\
namespace std {\
	inline S abs(const T A)	{\
		return std::sqrt(A.x * A.x + A.y * A.y);\
	}\
}

GEN(float, float_float)
GEN(double, double_double)
#undef GEN

namespace std {
	inline uint8_t	abs(const uint8_t x)	{	return x;	}
	inline uint16_t	abs(const uint16_t x)	{	return x;	}
	inline uint32_t	abs(const uint32_t x)	{	return x;	}
	inline uint64_t	abs(const uint64_t x)	{	return x;	}
}
