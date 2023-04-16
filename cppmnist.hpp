#ifndef CPPMNIST_HPP
#define CPPMNIST_HPP

#include <bit>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <type_traits>
#include <concepts>

namespace MNIST{

const std::uint8_t threshold = 128u;

std::uint32_t swapendian(std::uint32_t x) {
	return x>>24 | (x<<8)&0x00ff0000 | (x>>8)&0x0000ff00 | x<<24;
}

template<typename T=std::uint8_t>
class Images{
private:
	std::vector<std::vector<std::vector<T>>> data_;
	const std::string filename_;
	std::uint32_t ftype_;
	std::uint32_t num_;
	std::uint32_t width_;
	std::uint32_t height_;
	std::ifstream ifs;
	void read_header();
	void read_payload();
public:
	const std::vector<std::vector<std::vector<T>>>& data();
	Images(std::string filename);
	~Images();
};

template<typename T>
Images<T>::Images(std::string filename): filename_(filename) {
	this->read_header();
	this->read_payload();
}

template<typename T>
Images<T>::~Images() {
	this->ifs.close();
}

template<typename T>
const std::vector<std::vector<std::vector<T>>>& Images<T>::data() {
	return this->data_;
}

template<typename T>
void Images<T>::read_header() {
	this->ifs = std::ifstream(this->filename_, std::ios::in | std::ios::binary);
	if (!this->ifs.is_open()) {
		std::stringstream ss;
		ss << "Failed to read \"" << this->filename_
		   << "\"." << std::endl;
		throw std::runtime_error(ss.str());
	}
	this->ifs.read((char*)&this->ftype_, sizeof(this->ftype_));
	this->ifs.read((char*)&this->num_, sizeof(this->num_));
	this->ifs.read((char*)&this->width_, sizeof(this->width_));
	this->ifs.read((char*)&this->height_, sizeof(this->height_));
	if constexpr(std::endian::native == std::endian::little) {
		this->ftype_ = swapendian(this->ftype_);
		this->num_ = swapendian(this->num_);
		this->width_ = swapendian(this->width_);
		this->height_ = swapendian(this->height_);
	}
	if (this->ftype_ != 0x803u) {
		std::stringstream ss;
		ss << "This file is not a MNIST images file." << std::endl;
		throw std::runtime_error(ss.str());
	}
}

template<typename T>
void Images<T>::read_payload() {
	this->data_.resize(this->num_);
	for (auto& image: this->data_) {
		image.resize(this->height_);
		for (auto& row: image) {
			row.resize(this->width_);
			for (auto& pixel: row) {
				std::uint8_t tmp;
				this->ifs.read((char*)&tmp, sizeof(tmp));
				if constexpr(std::is_integral_v<T>) {
					pixel = tmp;
				}else {
					pixel = (T)tmp/255;
				}
			}
		}
	}
}

template<>
void Images<bool>::read_payload() {
	this->data_.resize(this->num_);
	for (auto& image: this->data_) {
		image.resize(this->height_);
		for (auto& row: image) {
			row.resize(this->width_);
			for (std::size_t i = 0; i < row.size(); i++) {
				std::uint8_t tmp;
				this->ifs.read((char*)&tmp, sizeof(tmp));
				row[i] = tmp >= threshold;
			}
		}
	}
}

template<typename T = int>
class Labels{
private:
	std::vector<T> data_;
	const std::string filename_;
	std::uint32_t ftype_;
	std::uint32_t num_;
	void read();
public:
	const std::vector<T>& data();
	Labels(std::string filename);
};

template<typename T>
Labels<T>::Labels(std::string filename): filename_(filename) {
	this->read();
}

template<typename T>
const std::vector<T>& Labels<T>::data() {
	return this->data_;
}

template<typename T>
void Labels<T>::read() {
	std::ifstream ifs(this->filename_, std::ios::in | std::ios::binary);
	if (!ifs.is_open()) {
		std::stringstream ss;
		ss << "Failed to read \"" << this->filename_
		   << "\"." << std::endl;
		throw std::runtime_error(ss.str());
	}
	ifs.read((char*)&this->ftype_, sizeof(this->ftype_));
	ifs.read((char*)&this->num_, sizeof(this->num_));
	if constexpr(std::endian::native == std::endian::little) {
		this->ftype_ = swapendian(this->ftype_);
		this->num_ = swapendian(this->num_);
	}
	if (this->ftype_ != 0x801u) {
		std::stringstream ss;
		ss << "This file is not a MNIST labels file." << std::endl;
		throw std::runtime_error(ss.str());
	}
	this->data_.resize(this->num_);
	for (auto& label: this->data_) {
		std::uint8_t tmp;
		ifs.read((char*)&tmp, sizeof(tmp));
		label = tmp;
	}
}

} // namespace mnist

#endif // include guard of CPPMNIST_HPP
