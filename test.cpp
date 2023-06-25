#include <iostream>
#include "cppmnist.hpp"

int main() {
	auto images = MNIST::Images<float>("../train-images-idx3-ubyte");
	auto labels = MNIST::Labels("../train-labels-idx1-ubyte");
	int cnt = 0;
	for (int idx = 0; idx < 100; idx++) {
		images.dump(idx, std::cout);
		int tmp;
		std::cout << idx << " out of " << 100 << std::endl;
		std::cin >> tmp;
		if (tmp == labels.basedata()[idx]) {
			std::cout << "Correct!" << std::endl;
			cnt++;
		}else {
			std::cout << "Wrong! The answer is "
					  << labels.basedata()[idx] << "." << std::endl;
		}
	}
	std::cout << cnt << std::endl;
	return 0;
}
