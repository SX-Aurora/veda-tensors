// Copyright 2022 NEC Laboratories Europe

#pragma once 

//------------------------------------------------------------------------------
#define CHECK_DIMS(A, B)	if(A != 1 && A != B)	FVEDA(VEDA_ERROR_NOT_IMPLEMENTED)

//------------------------------------------------------------------------------
template<typename TO, typename TX, typename OP>
inline void veda_tensors_x(TO* _o, const TX* _x, const size_t co, const size_t cx, const OP op) {
	CHECK_DIMS(cx, co);

	if(cx == 1) {
		auto x = *_x;
		auto value = op(x);
		veda_omp_simd(co, [&](const size_t min, const size_t max) {
			auto cnt = max - min;
			auto o = _o + min;
			for(auto i = 0; i < cnt; i++)
				o[i] = value;
		}, veda_omp_vlen<TO>());
	} else {
		veda_omp_simd(co, [&](const size_t min, const size_t max) {
			auto cnt = max - min;
			auto o = _o + min;
			auto x = _x + min;
			for(auto i = 0; i < cnt; i++) {
				o[i] = op(x[i]);
			}
		}, veda_omp_vlen<TO>());
	}
}

//------------------------------------------------------------------------------
template<typename TO, typename TX, typename TY, typename Op>
inline void veda_tensors_xy(TO* _o, const TX* _x, const TY* _y, const size_t co, const size_t cx, const size_t cy, const Op op) {
	CHECK_DIMS(cx, co);
	CHECK_DIMS(cy, co);

	if(cx == 1) {
		auto x = *_x;
		if(cy == 1) {
			auto y = *_y;
			auto value = op(x, y);
			veda_omp_simd(co, [&](const size_t min, const size_t max) {
				auto cnt = max - min;
				auto o = _o + min;
				for(auto i = 0; i < cnt; i++)
					o[i] = value;
			}, veda_omp_vlen<TO>());
		} else {
			veda_omp_simd(co, [&](const size_t min, const size_t max) {
				auto cnt = max - min;
				auto o = _o + min;
				auto y = _y + min;
				for(auto i = 0; i < cnt; i++) {
					o[i] = op(x, y[i]);
				}
			}, veda_omp_vlen<TO>());
		}
	} else {
		if(cy == 1) {
			auto y = *_y;
			veda_omp_simd(co, [&](const size_t min, const size_t max) {
				auto cnt = max - min;
				auto o = _o + min;
				auto x = _x + min;
				for(auto i = 0; i < cnt; i++) {
					o[i] = op(x[i], y);
				}
			}, veda_omp_vlen<TO>());
		} else {
			veda_omp_simd(co, [&](const size_t min, const size_t max) {
				auto cnt = max - min;
				auto o = _o + min;
				auto x = _x + min;
				auto y = _y + min;
				for(auto i = 0; i < cnt; i++) {
					o[i] = op(x[i], y[i]);
				}
			}, veda_omp_vlen<TO>());
		}
	}
}

//------------------------------------------------------------------------------
template<typename TO, typename TX, typename TY, typename TZ, typename Op>
inline void veda_tensors_xyz(TO* _o, const TX* _x, const TY* _y, const TZ* _z, const size_t co, const size_t cx, const size_t cy, const size_t cz, const Op op) {
	CHECK_DIMS(cx, co);
	CHECK_DIMS(cy, co);
	CHECK_DIMS(cz, co);

	if(cx == 1) {
		auto x = *_x;
		if(cy == 1) {
			auto y = *_y;
			if(cz == 1) {
				auto z = *_z;
				auto value = op(x, y, z);
				veda_omp_simd(co, [&](const size_t min, const size_t max) {
					auto cnt = max - min;
					auto o = _o + min;
					for(auto i = 0; i < cnt; i++)
						o[i] = value;
				}, veda_omp_vlen<TO>());
			} else {
				veda_omp_simd(co, [&](const size_t min, const size_t max) {
					auto cnt = max - min;
					auto o = _o + min;
					auto z = _z + min;
					for(auto i = 0; i < cnt; i++) {
						o[i] = op(x, y, z[i]);
					}
				}, veda_omp_vlen<TO>());
			}
		} else {
			if(cz == 1) {
				auto z = *_z;
				veda_omp_simd(co, [&](const size_t min, const size_t max) {
					auto cnt = max - min;
					auto o = _o + min;
					auto y = _y + min;
					for(auto i = 0; i < cnt; i++) {
						o[i] = op(x, y[i], z);
					}
				}, veda_omp_vlen<TO>());
			} else {
				veda_omp_simd(co, [&](const size_t min, const size_t max) {
					auto cnt = max - min;
					auto o = _o + min;
					auto y = _y + min;
					auto z = _z + min;
					for(auto i = 0; i < cnt; i++) {
						o[i] = op(x, y[i], z[i]);
					}
				}, veda_omp_vlen<TO>());
			}
		}
	} else {
		if(cy == 1) {
			auto y = *_y;
			if(cz == 1) {
				auto z = *_z;
				veda_omp_simd(co, [&](const size_t min, const size_t max) {
					auto cnt = max - min;
					auto o = _o + min;
					auto x = _x + min;
					for(auto i = 0; i < cnt; i++) {
						o[i] = op(x[i], y, z);
					}
				}, veda_omp_vlen<TO>());
			} else {
				veda_omp_simd(co, [&](const size_t min, const size_t max) {
					auto cnt = max - min;
					auto o = _o + min;
					auto x = _x + min;
					auto z = _z + min;
					for(auto i = 0; i < cnt; i++) {
						o[i] = op(x[i], y, z[i]);
					}
				}, veda_omp_vlen<TO>());
			}
		} else {
			if(cz == 1) {
				auto z = *_z;
				veda_omp_simd(co, [&](const size_t min, const size_t max) {
					auto cnt = max - min;
					auto o = _o + min;
					auto x = _x + min;
					auto y = _y + min;
					for(auto i = 0; i < cnt; i++) {
						o[i] = op(x[i], y[i], z);
					}
				}, veda_omp_vlen<TO>());
			} else {
				veda_omp_simd(co, [&](const size_t min, const size_t max) {
					auto cnt = max - min;
					auto o = _o + min;
					auto x = _x + min;
					auto y = _y + min;
					auto z = _z + min;
					for(auto i = 0; i < cnt; i++) {
						o[i] = op(x[i], y[i], z[i]);
					}
				}, veda_omp_vlen<TO>());
			}
		}
	}
}

//------------------------------------------------------------------------------
template<typename TO, typename TX, typename TY, typename TZ, typename TW, typename Op>
inline void veda_tensors_xyzw(TO* _o, const TX* _x, const TY* _y, const TZ* _z, const TW* _w, const size_t co, const size_t cx, const size_t cy, const size_t cz, const size_t cw, const Op op) {
	CHECK_DIMS(cx, co);
	CHECK_DIMS(cy, co);
	CHECK_DIMS(cz, co);
	CHECK_DIMS(cw, co);

	if(cx == 1) {
		auto x = *_x;
		if(cy == 1) {
			auto y = *_y;
			if(cz == 1) {
				auto z = *_z;
				if(cw == 1) {
					auto w = *_w;
					auto value = op(x, y, z, w);
					veda_omp_simd(co, [&](const size_t min, const size_t max) {
						auto cnt = max - min;
						auto o = _o + min;
						for(auto i = 0; i < cnt; i++)
							o[i] = value;
					}, veda_omp_vlen<TO>());
				} else {
					veda_omp_simd(co, [&](const size_t min, const size_t max) {
						auto cnt = max - min;
						auto o = _o + min;
						auto w = _w + min;
						for(auto i = 0; i < cnt; i++) {
							o[i] = op(x, y, z, w[i]);
						}
					}, veda_omp_vlen<TO>());
				}
			} else {
				if(cw == 1) {
					auto w = *_w;
					veda_omp_simd(co, [&](const size_t min, const size_t max) {
						auto cnt = max - min;
						auto o = _o + min;
						auto z = _z + min;
						for(auto i = 0; i < cnt; i++) {
							o[i] = op(x, y, z[i], w);
						}
					}, veda_omp_vlen<TO>());
				} else {
					veda_omp_simd(co, [&](const size_t min, const size_t max) {
						auto cnt = max - min;
						auto o = _o + min;
						auto z = _z + min;
						auto w = _w + min;
						for(auto i = 0; i < cnt; i++) {
							o[i] = op(x, y, z[i], w[i]);
						}
					}, veda_omp_vlen<TO>());
				}
			}
		} else {
			if(cz == 1) {
				auto z = *_z;
				if(cw == 1) {
					auto w = *_w;
					veda_omp_simd(co, [&](const size_t min, const size_t max) {
						auto cnt = max - min;
						auto o = _o + min;
						auto y = _y + min;
						for(auto i = 0; i < cnt; i++) {
							o[i] = op(x, y[i], z, w);
						}
					}, veda_omp_vlen<TO>());
				} else {
					veda_omp_simd(co, [&](const size_t min, const size_t max) {
						auto cnt = max - min;
						auto o = _o + min;
						auto y = _y + min;
						auto w = _w + min;
						for(auto i = 0; i < cnt; i++) {
							o[i] = op(x, y[i], z, w[i]);
						}
					}, veda_omp_vlen<TO>());
				}
			} else {
				if(cw == 1) {
					auto w = *_w;
					veda_omp_simd(co, [&](const size_t min, const size_t max) {
						auto cnt = max - min;
						auto o = _o + min;
						auto y = _y + min;
						auto z = _z + min;
						for(auto i = 0; i < cnt; i++) {
							o[i] = op(x, y[i], z[i], w);
						}
					}, veda_omp_vlen<TO>());
				} else {
					veda_omp_simd(co, [&](const size_t min, const size_t max) {
						auto cnt = max - min;
						auto o = _o + min;
						auto y = _y + min;
						auto z = _z + min;
						auto w = _w + min;
						for(auto i = 0; i < cnt; i++) {
							o[i] = op(x, y[i], z[i], w[i]);
						}
					}, veda_omp_vlen<TO>());
				}
			}
		}
	} else {
		if(cy == 1) {
			auto y = *_y;
			if(cz == 1) {
				auto z = *_z;
				if(cw == 1) {
					auto w = *_w;
					veda_omp_simd(co, [&](const size_t min, const size_t max) {
						auto cnt = max - min;
						auto o = _o + min;
						auto x = _x + min;
						for(auto i = 0; i < cnt; i++) {
							o[i] = op(x[i], y, z, w);
						}
					}, veda_omp_vlen<TO>());
				} else {
					veda_omp_simd(co, [&](const size_t min, const size_t max) {
						auto cnt = max - min;
						auto o = _o + min;
						auto x = _x + min;
						auto w = _w + min;
						for(auto i = 0; i < cnt; i++) {
							o[i] = op(x[i], y, z, w[i]);
						}
					}, veda_omp_vlen<TO>());
				}
			} else {
				if(cw == 1) {
					auto w = *_w;
					veda_omp_simd(co, [&](const size_t min, const size_t max) {
						auto cnt = max - min;
						auto o = _o + min;
						auto x = _x + min;
						auto z = _z + min;
						for(auto i = 0; i < cnt; i++) {
							o[i] = op(x[i], y, z[i], w);
						}
					}, veda_omp_vlen<TO>());
				} else {
					veda_omp_simd(co, [&](const size_t min, const size_t max) {
						auto cnt = max - min;
						auto o = _o + min;
						auto x = _x + min;
						auto z = _z + min;
						auto w = _w + min;
						for(auto i = 0; i < cnt; i++) {
							o[i] = op(x[i], y, z[i], w[i]);
						}
					}, veda_omp_vlen<TO>());
				}
			}
		} else {
			if(cz == 1) {
				auto z = *_z;
				if(cw == 1) {
					auto w = *_w;
					veda_omp_simd(co, [&](const size_t min, const size_t max) {
						auto cnt = max - min;
						auto o = _o + min;
						auto x = _x + min;
						auto y = _y + min;
						for(auto i = 0; i < cnt; i++) {
							o[i] = op(x[i], y[i], z, w);
						}
					}, veda_omp_vlen<TO>());
				} else {
					veda_omp_simd(co, [&](const size_t min, const size_t max) {
						auto cnt = max - min;
						auto o = _o + min;
						auto x = _x + min;
						auto y = _y + min;
						auto w = _w + min;
						for(auto i = 0; i < cnt; i++) {
							o[i] = op(x[i], y[i], z, w[i]);
						}
					}, veda_omp_vlen<TO>());
				}
			} else {
				if(cw == 1) {
					auto w = *_w;
					veda_omp_simd(co, [&](const size_t min, const size_t max) {
						auto cnt = max - min;
						auto o = _o + min;
						auto x = _x + min;
						auto y = _y + min;
						auto z = _z + min;
						for(auto i = 0; i < cnt; i++) {
							o[i] = op(x[i], y[i], z[i], w);
						}
					}, veda_omp_vlen<TO>());
				} else {
					veda_omp_simd(co, [&](const size_t min, const size_t max) {
						auto cnt = max - min;
						auto o = _o + min;
						auto x = _x + min;
						auto y = _y + min;
						auto z = _z + min;
						auto w = _w + min;
						for(auto i = 0; i < cnt; i++) {
							o[i] = op(x[i], y[i], z[i], w[i]);
						}
					}, veda_omp_vlen<TO>());
				}
			}
		}
	}
}

//------------------------------------------------------------------------------
#undef CHECK_DIMS
