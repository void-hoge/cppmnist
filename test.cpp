#include "cppmnist.hpp"

int main() {
	auto images = MNIST::Images<bool>("../train-images-idx3-ubyte");
	auto labels = MNIST::Labels("../train-labels-idx1-ubyte");
	auto onehot = labels.onehot();
	int cnt = 0;
	for (int idx = 0; idx < 100; idx++) {
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
		int tmp;
		std::cout << idx << " out of " << 100 << std::endl;
		std::cin >> tmp;
		if (tmp == labels.data()[idx]) {
			std::cout << "Correct!" << std::endl;
			cnt++;
		}else {
			std::cout << "Wrong! The answer is "
					  << labels.data()[idx] << "." << std::endl;
		}
	}
	std::cout << cnt << std::endl;
	return 0;
}
