// Copyright 2022-2025 NEC Laboratories Europe

#pragma once

inline uint64_t veda_tensors_time_us(void) {
	auto duration = std::chrono::system_clock::now().time_since_epoch();
	return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

class Profiler {
	const char*		m_name;
	const uint64_t	m_start;

public:
	inline Profiler(const char* name) : m_name(name), m_start(veda_tensors_time_us()) {
	}

	inline ~Profiler(void) {
		printf("%s: %.3fms\n", m_name, (veda_tensors_time_us()-m_start)/1000.0f);
	}
};

#define PROFILE() Profiler __profiler__(__PRETTY_FUNCTION__);
