#include <iostream>

#include "simd.hpp"

using namespace simd;

int main(int argc, char **argv) {
	m128 x, y, z;
	alignas(16) float buf[8];

	// read x in
	std::cout << "x = " << std::endl;
	std::cin >> buf[0] >> buf[1] >> buf[2] >> buf[3];

	// read y in
	std::cout << "y = " << std::endl;
	std::cin >> buf[4] >> buf[5] >> buf[6] >> buf[7];

	x = load_m128_aligned(buf);
	y = load_m128_aligned(buf + 4);
	z = shufps<0, 2, 1, 3>(x, y); // z = { x[0], x[2], y[1], y[3] }
	z = addps(z, y);
	z = mulps(x, z);
	z = sqrtps(z);

	// output
    std::cout << z[0] << ", " << z[1] << ", " << z[2] << ", " << z[3] << std::endl;	

    return 0;
}
