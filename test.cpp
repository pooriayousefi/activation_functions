
#include <iostream>
#include <stdexcept>
#include "activation_functions.h"

// entry point
auto main()->int
{
	try
	{
		std::cout
			<< "\n\tidentity(-0.25f) = " << activation_function::identity(-0.25f)
			<< "\n\tbinary step(-3) = " << activation_function::binary_step(-3)
			<< "\n\tsaturating linear(0.03) = " << activation_function::saturating_linear(0.03)
			<< "\n\tsaturating linear(0ULL) = " << activation_function::saturating_linear(0ULL)
			<< "\n\tsaturating linear(2LL) = " << activation_function::saturating_linear(2LL)
			<< "\n\tgeneralized logistic(-2.3f, 5) = " << activation_function::generalized_logistic(-2.3f, 5)
			<< "\n\tSoboleva modified hyperbolic tangent(1.1, -2L, 5, 0.1f, 4.5e-4) = " << activation_function::Soboleva_modified_hyperbolic_tangent(1.1, -2L, 5, 0.1f, 4.5e-4)
			<< "\n\tReLU(-2.3) = " << activation_function::ReLU(-2.3)
			<< "\n\tSiL(-2.3e-9, 1.18e-4) = " << activation_function::SiL(-2.3e-9, 1.18e-4)
			<< "\n\terror function(0.25) = " << activation_function::error_function(0.25)
			<< "\n\tGudermanian(1.18e4) = " << activation_function::Gudermanian(1.18e4)
			<< "\n\n";

		return EXIT_SUCCESS;
	}
	catch (const std::exception& xxx)
	{
		std::cerr << xxx.what() << std::endl;
		return EXIT_FAILURE;
	}
}
