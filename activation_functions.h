
#pragma once
#include <execution>
#include <concepts>
#include <algorithm>
#include <cmath>

namespace
{
	// ------------------------------------------------
	//
	// 
	//          yet another concepts and types
	//
	// 
	// ------------------------------------------------

	template<typename T>
	concept arithmetic_value = std::integral<T> || std::floating_point<T>;

	template<arithmetic_value... T>
	using real_value = std::common_type_t<float, T...>;

	// ------------------------------------------------
	//                     
	//                     
	//               activation functions
	//                      
	//                      
	// ------------------------------------------------
	namespace activation_function
	{
		// identity function
		template<arithmetic_value X>
		auto identity(X x)->X
		{
			return x;
		}

		// binary step function
		template<arithmetic_value X>
		auto binary_step(X x)->X
		{
			return (x < (X)0 ? (X)0 : (X)1);
		}

		// hard limiter function
		template<arithmetic_value X>
		auto hard_limiter(X x)->X
		{
			return (x < (X)0 ? (X)0 : (X)1);
		}

		// saturating linear function
		template<arithmetic_value X>
		auto saturating_linear(X x)->X
		{
			auto y{ (X)0 };
			if (x >= (X)0 && x <= (X)1)
				y = x;
			else if (x > (X)1)
				y = (X)1;
			return y;
		}

		// logistic function
		template<arithmetic_value X>
		auto logistic(X x)->real_value<X>
		{
			using T = real_value<X>;
			return (1 / (1 + exp(-(T)x)));
		}

		// soft step function
		template<arithmetic_value X>
		auto soft_step(X x)->real_value<X>
		{
			using T = real_value<X>;
			return (1 / (1 + exp(-(T)x)));
		}

		// generalized logistic function
		template<arithmetic_value X, arithmetic_value P>
		auto generalized_logistic(X x, P p)->real_value<X, P>
		{
			using T = real_value<X, P>;
			return pow(1 + exp(-(T)x), -p);
		}

		// hyperbolic tangent function
		template<arithmetic_value X>
		auto hyperbolic_tangent(X x)->real_value<X>
		{
			using T = real_value<X>;
			return tanh((T)x);
		}

		// Soboleva modified hyperbolic function
		template<arithmetic_value X, arithmetic_value A, arithmetic_value B, arithmetic_value C, arithmetic_value D>
		auto Soboleva_modified_hyperbolic_tangent(X x, A a, B b, C c, D d)->real_value<X, A, B, C, D>
		{
			using T = real_value<X, A, B, C, D>;
			return (exp(a * (T)x) - exp(-b * (T)x)) / (exp(c * (T)x) + exp(-d * (T)x));
		}

		// modified hyperbolic function = Soboleva modified hyperbolic function
		template<arithmetic_value X, arithmetic_value A, arithmetic_value B, arithmetic_value C, arithmetic_value D>
		auto mtanh(X x, A a, B b, C c, D d)->real_value<X, A, B, C, D>
		{
			using T = real_value<X, A, B, C, D>;
			return (exp(a * (T)x) - exp(-b * (T)x)) / (exp(c * (T)x) + exp(-d * (T)x));
		}

		// arctangent function
		template<arithmetic_value X>
		auto arctangent(X x)->real_value<X>
		{
			using T = real_value<X>;
			return atan((T)x);
		}

		// Gudermanian function
		template<arithmetic_value X>
		auto Gudermanian(X x)->real_value<X>
		{
			using T = real_value<X>;
			return 2 * atan(tanh((T)x / 2));
		}

		// error function
		template<arithmetic_value X>
		auto error_function(X x)->real_value<X>
		{
			using T = real_value<X>;
			return erf((T)x);
		}

		// swish function
		template<arithmetic_value X, arithmetic_value B>
		auto swish(X x, B b)->real_value<X, B>
		{
			using T = real_value<X, B>;
			return (T)x / (1 + exp(-b * (T)x));
		}

		// linear unit function = swish sunction
		template<arithmetic_value X, arithmetic_value B>
		auto linear_unit(X x, B b)->real_value<X, B>
		{
			using T = real_value<X, B>;
			return (T)x / (1 + exp(-b * (T)x));
		}

		// SiLU function = swish function
		template<arithmetic_value X, arithmetic_value B>
		auto SiLU(X x, B b)->real_value<X, B>
		{
			using T = real_value<X, B>;
			return (T)x / (1 + exp(-b * (T)x));
		}

		// shrinkage function = swish function
		template<arithmetic_value X, arithmetic_value B>
		auto shrinkage(X x, B b)->real_value<X, B>
		{
			using T = real_value<X, B>;
			return (T)x / (1 + exp(-b * (T)x));
		}

		// SiL function = swish function
		template<arithmetic_value X, arithmetic_value B>
		auto SiL(X x, B b)->real_value<X, B>
		{
			using T = real_value<X, B>;
			return (T)x / (1 + exp(-b * (T)x));
		}

		// rectified linear unit function
		template<arithmetic_value X>
		auto rectified_linear_unit(X x)->X
		{
			return std::max((X)0, x);
		}

		// ReLU function
		template<arithmetic_value X>
		auto ReLU(X x)->X
		{
			return std::max((X)0, x);
		}
	}
}
