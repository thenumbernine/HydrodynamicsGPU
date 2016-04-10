#pragma once

//used by enough folks:
//don't know where to put it ...

template<typename T> std::string toNumericString(T value);

template<> inline std::string toNumericString<double>(double value) {
	std::string s = std::to_string((long double)value);
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
