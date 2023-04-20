#include "cppmnist.hpp"
#include <iostream>

int main() {
	auto images = MNIST::Images<double>("../train-images-idx3-ubyte");
	auto labels = MNIST::Labels("../train-labels-idx1-ubyte");
	auto onehot = labels.onehot();
	int cnt = 0;
	for (int idx = 0; idx < 100; idx++) {
		images.dump(idx, std::cout);
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
