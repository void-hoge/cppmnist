#include "cppmnist.hpp"

int main() {
	auto images = MNIST::Images<bool>("../t10k-images-idx3-ubyte");
	auto labels = MNIST::Labels("../t10k-labels-idx1-ubyte");
	for (int idx = 0; idx < 10; idx++) {
		for (const auto& row: images.data()[idx]) {
			for (const auto& pixel: row) {
				if (pixel == false) {
					std::cout << "  ";
				}else {
					std::cout << "xx";
				}
			}
			std::cout << std::endl;
		}
		std::cout << labels.data()[idx] << std::endl;
	}
	return 0;
}
