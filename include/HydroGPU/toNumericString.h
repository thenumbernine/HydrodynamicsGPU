#pragma once

#include <sstream>

/*
serializing ints to OpenCL can safely use std::to_string
for converting float/double, use this:
*/

template<typename T> std::string toNumericString(T value);

template<> inline std::string toNumericString<double>(double value) {
	std::stringstream ss;
	ss << value;
	std::string s = ss.str();
	if (s.find("e") == std::string::npos) {
		if (s.find(".") == std::string::npos) {
			s += ".";
		}
	}
	return s;
}

template<> inline std::string toNumericString<float>(float value) {
	return toNumericString<double>(value) + "f";
}
