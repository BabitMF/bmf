/*
 * Copyright 2023 Babit Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// CUDA-device-accessible version of gcc 4.9.3's <limits>. -*- C++ -*-

/*
 * The C++ standard library's numeric_limits classes cannot currently be used
 * in device code, since their methods are not marked HMP_HOST_DEVICE, and
 * even though they're constexpr - CUDA 7.5 and earlier does not properly support
 * using  __host__ constpexr's on the device  (there's experimental support but
 * it causes trouble in different places).
 */

// Copyright (C) 1999-2014 Free Software Foundation, Inc.
//
// This file is a modification of part of
// GNU ISO C++ Library.  It is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This file is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// Under Section 7 of GPL version 3, you are granted additional
// permissions described in the GCC Runtime Library Exception, version
// 3.1, as published by the Free Software Foundation.

// See the GNU GPL and the GCC Runtime Library Exception at
// <http://www.gnu.org/licenses/>.
#pragma once
#ifndef ON_DEVICE_GLIBC_LIKE_NUMERIC_LIMITS_H_
#define ON_DEVICE_GLIBC_LIKE_NUMERIC_LIMITS_H_ 1

// We had probably better just make some assumptions about nvcc's
// target rather than take the host-related flags from here:
//#include <bits/c++config.h>
#include <hmp/core/half.h>

//
// The numeric_limits<> traits document implementation-defined aspects
// of fundamental arithmetic data types (integers and floating points).
// From Standard C++ point of view, there are 14 such types:
//   * integers
//         bool							(1)
//         char, signed char, unsigned char, wchar_t            (4)
//         short, unsigned short				(2)
//         int, unsigned					(2)
//         long, unsigned long					(2)
//
//   * floating points
//         float						(1)
//         double						(1)
//         long double						(1)
//
// GNU C++ understands (where supported by the host C-library)
//   * integer
//         long long, unsigned long long			(2)
//
// which brings us to 16 fundamental arithmetic data types in GNU C++.
//
//
// Since a numeric_limits<> is a bit tricky to get right, we rely on
// an interface composed of macros which should be defined in config/os
// or config/cpu when they differ from the generic (read arbitrary)
// definitions given here.
//

// These values can be overridden in the target configuration file.
// The default values are appropriate for many 32-bit targets.

// GCC only intrinsically supports modulo integral types.  The only remaining
// integral exceptional values is division by zero.  Only targets that do not
// signal division by zero in some "hard to ignore" way should use false.
#ifndef __glibcxx_integral_traps
# define __glibcxx_integral_traps true
#endif

// float
//

// Default values.  Should be overridden in configuration files if necessary.

#ifndef __glibcxx_float_has_denorm_loss
#  define __glibcxx_float_has_denorm_loss false
#endif
#ifndef __glibcxx_float_traps
#  define __glibcxx_float_traps false
#endif
#ifndef __glibcxx_float_tinyness_before
#  define __glibcxx_float_tinyness_before false
#endif

// double

// Default values.  Should be overridden in configuration files if necessary.

#ifndef __glibcxx_double_has_denorm_loss
#  define __glibcxx_double_has_denorm_loss false
#endif
#ifndef __glibcxx_double_traps
#  define __glibcxx_double_traps false
#endif
#ifndef __glibcxx_double_tinyness_before
#  define __glibcxx_double_tinyness_before false
#endif

// long double

// Default values.  Should be overridden in configuration files if necessary.

#ifndef __glibcxx_long_double_has_denorm_loss
#  define __glibcxx_long_double_has_denorm_loss false
#endif
#ifndef __glibcxx_long_double_traps
#  define __glibcxx_long_double_traps false
#endif
#ifndef __glibcxx_long_double_tinyness_before
#  define __glibcxx_long_double_tinyness_before false
#endif

// You should not need to define any macros below this point.

#define __glibcxx_signed(T)	((T)(-1) < 0)

#define __glibcxx_min(T) \
  (__glibcxx_signed (T) ? -__glibcxx_max (T) - 1 : (T)0)

#define __glibcxx_max(T) \
  (__glibcxx_signed (T) ? \
   (((((T)1 << (__glibcxx_digits (T) - 1)) - 1) << 1) + 1) : ~(T)0)

#define __glibcxx_digits(T) \
  (sizeof(T) * __CHAR_BIT__ - __glibcxx_signed (T))

// The fraction 643/2136 approximates log10(2) to 7 significant digits.
#define __glibcxx_digits10(T) \
  (__glibcxx_digits (T) * 643L / 2136)

#define __glibcxx_max_digits10(T) \
  (2 + (T) * 643L / 2136)

namespace hmp
{
  /**
   *  @brief Describes the rounding style for floating-point types.
   *
   *  This is used in the std::numeric_limits class.
  */
  enum float_round_style
  {
    round_indeterminate       = -1,    /// Intermediate.
    round_toward_zero         = 0,     /// To zero.
    round_to_nearest          = 1,     /// To the nearest representable value.
    round_toward_infinity     = 2,     /// To infinity.
    round_toward_neg_infinity = 3      /// To negative infinity.
  };

  /**
   *  @brief Describes the denormalization for floating-point types.
   *
   *  These values represent the presence or absence of a variable number
   *  of exponent bits.  This type is used in the std::numeric_limits class.
  */
  enum float_denorm_style
  {
    /// Indeterminate at compile time whether denormalized values are allowed.
    denorm_indeterminate = -1,
    /// The type does not allow denormalized values.
    denorm_absent        = 0,
    /// The type allows denormalized values.
    denorm_present       = 1
  };

  /**
   *  @brief Part of std::numeric_limits.
   *
   *  The @c static @c const members are usable as integral constant
   *  expressions.
   *
   *  @note This is a separate class for purposes of efficiency; you
   *        should only access these members as part of an instantiation
   *        of the std::numeric_limits class.
  */
  struct __numeric_limits_base
  {
    /** This will be true for all fundamental types (which have
	specializations), and false for everything else.  */
    static constexpr bool is_specialized = false;

    /** The number of @c radix digits that be represented without change:  for
	integer types, the number of non-sign bits in the mantissa; for
	floating types, the number of @c radix digits in the mantissa.  */
    static constexpr int digits = 0;

    /** The number of base 10 digits that can be represented without change. */
    static constexpr int digits10 = 0;

    /** The number of base 10 digits required to ensure that values which
	differ are always differentiated.  */
    static constexpr int max_digits10 = 0;

    /** True if the type is signed.  */
    static constexpr bool is_signed = false;

    /** True if the type is integer.  */
    static constexpr bool is_integer = false;

    /** True if the type uses an exact representation. All integer types are
	exact, but not all exact types are integer.  For example, rational and
	fixed-exponent representations are exact but not integer. */
    static constexpr bool is_exact = false;

    /** For integer types, specifies the base of the representation.  For
	floating types, specifies the base of the exponent representation.  */
    static constexpr int radix = 0;

    /** The minimum negative integer such that @c radix raised to the power of
	(one less than that integer) is a normalized floating point number.  */
    static constexpr int min_exponent = 0;

    /** The minimum negative integer such that 10 raised to that power is in
	the range of normalized floating point numbers.  */
    static constexpr int min_exponent10 = 0;

    /** The maximum positive integer such that @c radix raised to the power of
	(one less than that integer) is a representable finite floating point
	number.  */
    static constexpr int max_exponent = 0;

    /** The maximum positive integer such that 10 raised to that power is in
	the range of representable finite floating point numbers.  */
    static constexpr int max_exponent10 = 0;

    /** True if the type has a representation for positive infinity.  */
    static constexpr bool has_infinity = false;

    /** True if the type has a representation for a quiet (non-signaling)
	Not a Number.  */
    static constexpr bool has_quiet_NaN = false;

    /** True if the type has a representation for a signaling
	Not a Number.  */
    static constexpr bool has_signaling_NaN = false;

    /** See std::float_denorm_style for more information.  */
    static constexpr float_denorm_style has_denorm = denorm_absent;

    /** True if loss of accuracy is detected as a denormalization loss,
	rather than as an inexact result. */
    static constexpr bool has_denorm_loss = false;

    /** True if-and-only-if the type adheres to the IEC 559 standard, also
	known as IEEE 754.  (Only makes sense for floating point types.)  */
    static constexpr bool is_iec559 = false;

    /** True if the set of values representable by the type is
	finite.  All built-in types are bounded, this member would be
	false for arbitrary precision types. */
    static constexpr bool is_bounded = false;

    /** True if the type is @e modulo. A type is modulo if, for any
	operation involving +, -, or * on values of that type whose
	result would fall outside the range [min(),max()], the value
	returned differs from the true value by an integer multiple of
	max() - min() + 1. On most machines, this is false for floating
	types, true for unsigned integers, and true for signed integers.
	See PR22200 about signed integers.  */
    static constexpr bool is_modulo = false;

    /** True if trapping is implemented for this type.  */
    static constexpr bool traps = false;

    /** True if tininess is detected before rounding.  (see IEC 559)  */
    static constexpr bool tinyness_before = false;

    /** See std::float_round_style for more information.  This is only
	meaningful for floating types; integer types will all be
	round_toward_zero.  */
    static constexpr float_round_style round_style =
						    round_toward_zero;
  };

  /**
   *  @brief Properties of fundamental types.
   *
   *  This class allows a program to obtain information about the
   *  representation of a fundamental type on a given platform.  For
   *  non-fundamental types, the functions will return 0 and the data
   *  members will all be @c false.
   *
   *  _GLIBCXX_RESOLVE_LIB_DEFECTS:  DRs 201 and 184 (hi Gaby!) are
   *  noted, but not incorporated in this documented (yet).
  */
  template<typename _Tp>
    struct numeric_limits : public __numeric_limits_base
    {
      /** The minimum finite value, or for floating types with
	  denormalization, the minimum positive normalized value.  */
       HMP_HOST_DEVICE static constexpr _Tp
      min() noexcept { return _Tp(); }

      /** The maximum finite value.  */
      HMP_HOST_DEVICE static constexpr _Tp
      max() noexcept { return _Tp(); }

      /** A finite value x such that there is no other finite value y
       *  where y < x.  */
      HMP_HOST_DEVICE static constexpr _Tp
      lowest() noexcept { return _Tp(); }

      /** The @e machine @e epsilon:  the difference between 1 and the least
	  value greater than 1 that is representable.  */
      HMP_HOST_DEVICE static constexpr _Tp
      epsilon() noexcept { return _Tp(); }

      /** The maximum rounding error measurement (see LIA-1).  */
      HMP_HOST_DEVICE static constexpr _Tp
      round_error() noexcept { return _Tp(); }

      /** The representation of positive infinity, if @c has_infinity.  */
      HMP_HOST_DEVICE static constexpr _Tp
      infinity() noexcept { return _Tp(); }

      /** The representation of a quiet Not a Number,
	  if @c has_quiet_NaN. */
      HMP_HOST_DEVICE static constexpr _Tp
      quiet_NaN() noexcept { return _Tp(); }

      /** The representation of a signaling Not a Number, if
	  @c has_signaling_NaN. */
      HMP_HOST_DEVICE static constexpr _Tp
      signaling_NaN() noexcept { return _Tp(); }

      /** The minimum positive denormalized value.  For types where
	  @c has_denorm is false, this is the minimum positive normalized
	  value.  */
      HMP_HOST_DEVICE static constexpr _Tp
      denorm_min() noexcept { return _Tp(); }
    };

  template<typename _Tp>
    struct numeric_limits<const _Tp>
    : public numeric_limits<_Tp> { };

  template<typename _Tp>
    struct numeric_limits<volatile _Tp>
    : public numeric_limits<_Tp> { };

  template<typename _Tp>
    struct numeric_limits<const volatile _Tp>
    : public numeric_limits<_Tp> { };

  // Now there follow 16 explicit specializations.  Yes, 16.  Make sure
  // you get the count right. (18 in c++0x mode)

  /// numeric_limits<bool> specialization.
  template<>
    struct numeric_limits<bool>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr bool
      min() noexcept { return false; }

      HMP_HOST_DEVICE static constexpr bool
      max() noexcept { return true; }

      HMP_HOST_DEVICE static constexpr bool
      lowest() noexcept { return min(); }
      static constexpr int digits = 1;
      static constexpr int digits10 = 0;
      static constexpr int max_digits10 = 0;
      static constexpr bool is_signed = false;
      static constexpr bool is_integer = true;
      static constexpr bool is_exact = true;
      static constexpr int radix = 2;

      HMP_HOST_DEVICE static constexpr bool
      epsilon() noexcept { return false; }

      HMP_HOST_DEVICE static constexpr bool
      round_error() noexcept { return false; }

      static constexpr int min_exponent = 0;
      static constexpr int min_exponent10 = 0;
      static constexpr int max_exponent = 0;
      static constexpr int max_exponent10 = 0;

      static constexpr bool has_infinity = false;
      static constexpr bool has_quiet_NaN = false;
      static constexpr bool has_signaling_NaN = false;
      static constexpr float_denorm_style has_denorm
       = denorm_absent;
      static constexpr bool has_denorm_loss = false;

      HMP_HOST_DEVICE static constexpr bool
      infinity() noexcept { return false; }

      HMP_HOST_DEVICE static constexpr bool
      quiet_NaN() noexcept { return false; }

      HMP_HOST_DEVICE static constexpr bool
      signaling_NaN() noexcept { return false; }

      HMP_HOST_DEVICE static constexpr bool
      denorm_min() noexcept { return false; }

      static constexpr bool is_iec559 = false;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = false;

      // It is not clear what it means for a boolean type to trap.
      // This is a DR on the LWG issue list.  Here, I use integer
      // promotion semantics.
      static constexpr bool traps = __glibcxx_integral_traps;
      static constexpr bool tinyness_before = false;
      static constexpr float_round_style round_style
       = round_toward_zero;
    };

  /// numeric_limits<char> specialization.
  template<>
    struct numeric_limits<char>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr char
      min() noexcept { return __glibcxx_min(char); }

      HMP_HOST_DEVICE static constexpr char
      max() noexcept { return __glibcxx_max(char); }

      HMP_HOST_DEVICE static constexpr char
      lowest() noexcept { return min(); }

      static constexpr int digits = __glibcxx_digits (char);
      static constexpr int digits10 = __glibcxx_digits10 (char);
      static constexpr int max_digits10 = 0;
      static constexpr bool is_signed = __glibcxx_signed (char);
      static constexpr bool is_integer = true;
      static constexpr bool is_exact = true;
      static constexpr int radix = 2;

      HMP_HOST_DEVICE static constexpr char
      epsilon() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr char
      round_error() noexcept { return 0; }

      static constexpr int min_exponent = 0;
      static constexpr int min_exponent10 = 0;
      static constexpr int max_exponent = 0;
      static constexpr int max_exponent10 = 0;

      static constexpr bool has_infinity = false;
      static constexpr bool has_quiet_NaN = false;
      static constexpr bool has_signaling_NaN = false;
      static constexpr float_denorm_style has_denorm
       = denorm_absent;
      static constexpr bool has_denorm_loss = false;

      HMP_HOST_DEVICE static constexpr
      char infinity() noexcept { return char(); }

      HMP_HOST_DEVICE static constexpr char
      quiet_NaN() noexcept { return char(); }

      HMP_HOST_DEVICE static constexpr char
      signaling_NaN() noexcept { return char(); }

      HMP_HOST_DEVICE static constexpr char
      denorm_min() noexcept { return static_cast<char>(0); }

      static constexpr bool is_iec559 = false;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = !is_signed;

      static constexpr bool traps = __glibcxx_integral_traps;
      static constexpr bool tinyness_before = false;
      static constexpr float_round_style round_style
       = round_toward_zero;
    };

  /// numeric_limits<signed char> specialization.
  template<>
    struct numeric_limits<signed char>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr signed char
      min() noexcept { return -__SCHAR_MAX__ - 1; }

      HMP_HOST_DEVICE static constexpr signed char
      max() noexcept { return __SCHAR_MAX__; }

      HMP_HOST_DEVICE static constexpr signed char
      lowest() noexcept { return min(); }


      static constexpr int digits = __glibcxx_digits (signed char);
      static constexpr int digits10
       = __glibcxx_digits10 (signed char);
      static constexpr int max_digits10 = 0;
      static constexpr bool is_signed = true;
      static constexpr bool is_integer = true;
      static constexpr bool is_exact = true;
      static constexpr int radix = 2;

      HMP_HOST_DEVICE static constexpr signed char
      epsilon() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr signed char
      round_error() noexcept { return 0; }

      static constexpr int min_exponent = 0;
      static constexpr int min_exponent10 = 0;
      static constexpr int max_exponent = 0;
      static constexpr int max_exponent10 = 0;

      static constexpr bool has_infinity = false;
      static constexpr bool has_quiet_NaN = false;
      static constexpr bool has_signaling_NaN = false;
      static constexpr float_denorm_style has_denorm
       = denorm_absent;
      static constexpr bool has_denorm_loss = false;

      HMP_HOST_DEVICE static constexpr signed char
      infinity() noexcept { return static_cast<signed char>(0); }

      HMP_HOST_DEVICE static constexpr signed char
      quiet_NaN() noexcept { return static_cast<signed char>(0); }

      HMP_HOST_DEVICE static constexpr signed char
      signaling_NaN() noexcept
      { return static_cast<signed char>(0); }

      HMP_HOST_DEVICE static constexpr signed char
      denorm_min() noexcept
      { return static_cast<signed char>(0); }

      static constexpr bool is_iec559 = false;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = false;

      static constexpr bool traps = __glibcxx_integral_traps;
      static constexpr bool tinyness_before = false;
      static constexpr float_round_style round_style
       = round_toward_zero;
    };

  /// numeric_limits<unsigned char> specialization.
  template<>
    struct numeric_limits<unsigned char>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr unsigned char
      min() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr unsigned char
      max() noexcept { return __SCHAR_MAX__ * 2U + 1; }

      HMP_HOST_DEVICE static constexpr unsigned char
      lowest() noexcept { return min(); }

      static constexpr int digits
       = __glibcxx_digits (unsigned char);
      static constexpr int digits10
       = __glibcxx_digits10 (unsigned char);
      static constexpr int max_digits10 = 0;

      static constexpr bool is_signed = false;
      static constexpr bool is_integer = true;
      static constexpr bool is_exact = true;
      static constexpr int radix = 2;

      HMP_HOST_DEVICE static constexpr unsigned char
      epsilon() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr unsigned char
      round_error() noexcept { return 0; }

      static constexpr int min_exponent = 0;
      static constexpr int min_exponent10 = 0;
      static constexpr int max_exponent = 0;
      static constexpr int max_exponent10 = 0;

      static constexpr bool has_infinity = false;
      static constexpr bool has_quiet_NaN = false;
      static constexpr bool has_signaling_NaN = false;
      static constexpr float_denorm_style has_denorm
       = denorm_absent;
      static constexpr bool has_denorm_loss = false;

      HMP_HOST_DEVICE static constexpr unsigned char
      infinity() noexcept
      { return static_cast<unsigned char>(0); }

      HMP_HOST_DEVICE static constexpr unsigned char
      quiet_NaN() noexcept
      { return static_cast<unsigned char>(0); }

      HMP_HOST_DEVICE static constexpr unsigned char
      signaling_NaN() noexcept
      { return static_cast<unsigned char>(0); }

      HMP_HOST_DEVICE static constexpr unsigned char
      denorm_min() noexcept
      { return static_cast<unsigned char>(0); }

      static constexpr bool is_iec559 = false;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = true;

      static constexpr bool traps = __glibcxx_integral_traps;
      static constexpr bool tinyness_before = false;
      static constexpr float_round_style round_style
       = round_toward_zero;
    };

  /// numeric_limits<wchar_t> specialization.
  template<>
    struct numeric_limits<wchar_t>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr wchar_t
      min() noexcept { return __glibcxx_min (wchar_t); }

      HMP_HOST_DEVICE static constexpr wchar_t
      max() noexcept { return __glibcxx_max (wchar_t); }

      HMP_HOST_DEVICE static constexpr wchar_t
      lowest() noexcept { return min(); }


      static constexpr int digits = __glibcxx_digits (wchar_t);
      static constexpr int digits10
       = __glibcxx_digits10 (wchar_t);
      static constexpr int max_digits10 = 0;

      static constexpr bool is_signed = __glibcxx_signed (wchar_t);
      static constexpr bool is_integer = true;
      static constexpr bool is_exact = true;
      static constexpr int radix = 2;

      HMP_HOST_DEVICE static constexpr wchar_t
      epsilon() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr wchar_t
      round_error() noexcept { return 0; }

      static constexpr int min_exponent = 0;
      static constexpr int min_exponent10 = 0;
      static constexpr int max_exponent = 0;
      static constexpr int max_exponent10 = 0;

      static constexpr bool has_infinity = false;
      static constexpr bool has_quiet_NaN = false;
      static constexpr bool has_signaling_NaN = false;
      static constexpr float_denorm_style has_denorm
       = denorm_absent;
      static constexpr bool has_denorm_loss = false;

      HMP_HOST_DEVICE static constexpr wchar_t
      infinity() noexcept { return wchar_t(); }

      HMP_HOST_DEVICE static constexpr wchar_t
      quiet_NaN() noexcept { return wchar_t(); }

      HMP_HOST_DEVICE static constexpr wchar_t
      signaling_NaN() noexcept { return wchar_t(); }

      HMP_HOST_DEVICE static constexpr wchar_t
      denorm_min() noexcept { return wchar_t(); }

      static constexpr bool is_iec559 = false;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = !is_signed;

      static constexpr bool traps = __glibcxx_integral_traps;
      static constexpr bool tinyness_before = false;
      static constexpr float_round_style round_style
       = round_toward_zero;
    };

#if __cplusplus >= 201103L
  /// numeric_limits<char16_t> specialization.
  template<>
    struct numeric_limits<char16_t>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr char16_t
      min() noexcept { return __glibcxx_min (char16_t); }

      HMP_HOST_DEVICE static constexpr char16_t
      max() noexcept { return __glibcxx_max (char16_t); }

      HMP_HOST_DEVICE static constexpr char16_t
      lowest() noexcept { return min(); }

      static constexpr int digits = __glibcxx_digits (char16_t);
      static constexpr int digits10 = __glibcxx_digits10 (char16_t);
      static constexpr int max_digits10 = 0;
      static constexpr bool is_signed = __glibcxx_signed (char16_t);
      static constexpr bool is_integer = true;
      static constexpr bool is_exact = true;
      static constexpr int radix = 2;

      HMP_HOST_DEVICE static constexpr char16_t
      epsilon() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr char16_t
      round_error() noexcept { return 0; }

      static constexpr int min_exponent = 0;
      static constexpr int min_exponent10 = 0;
      static constexpr int max_exponent = 0;
      static constexpr int max_exponent10 = 0;

      static constexpr bool has_infinity = false;
      static constexpr bool has_quiet_NaN = false;
      static constexpr bool has_signaling_NaN = false;
      static constexpr float_denorm_style has_denorm = denorm_absent;
      static constexpr bool has_denorm_loss = false;

      HMP_HOST_DEVICE static constexpr char16_t
      infinity() noexcept { return char16_t(); }

      HMP_HOST_DEVICE static constexpr char16_t
      quiet_NaN() noexcept { return char16_t(); }

      HMP_HOST_DEVICE static constexpr char16_t
      signaling_NaN() noexcept { return char16_t(); }

      HMP_HOST_DEVICE static constexpr char16_t
      denorm_min() noexcept { return char16_t(); }

      static constexpr bool is_iec559 = false;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = !is_signed;

      static constexpr bool traps = __glibcxx_integral_traps;
      static constexpr bool tinyness_before = false;
      static constexpr float_round_style round_style = round_toward_zero;
    };

  /// numeric_limits<char32_t> specialization.
  template<>
    struct numeric_limits<char32_t>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr char32_t
      min() noexcept { return __glibcxx_min (char32_t); }

      HMP_HOST_DEVICE static constexpr char32_t
      max() noexcept { return __glibcxx_max (char32_t); }

      HMP_HOST_DEVICE static constexpr char32_t
      lowest() noexcept { return min(); }

      static constexpr int digits = __glibcxx_digits (char32_t);
      static constexpr int digits10 = __glibcxx_digits10 (char32_t);
      static constexpr int max_digits10 = 0;
      static constexpr bool is_signed = __glibcxx_signed (char32_t);
      static constexpr bool is_integer = true;
      static constexpr bool is_exact = true;
      static constexpr int radix = 2;

      HMP_HOST_DEVICE static constexpr char32_t
      epsilon() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr char32_t
      round_error() noexcept { return 0; }

      static constexpr int min_exponent = 0;
      static constexpr int min_exponent10 = 0;
      static constexpr int max_exponent = 0;
      static constexpr int max_exponent10 = 0;

      static constexpr bool has_infinity = false;
      static constexpr bool has_quiet_NaN = false;
      static constexpr bool has_signaling_NaN = false;
      static constexpr float_denorm_style has_denorm = denorm_absent;
      static constexpr bool has_denorm_loss = false;

      HMP_HOST_DEVICE static constexpr char32_t
      infinity() noexcept { return char32_t(); }

      HMP_HOST_DEVICE static constexpr char32_t
      quiet_NaN() noexcept { return char32_t(); }

      HMP_HOST_DEVICE static constexpr char32_t
      signaling_NaN() noexcept { return char32_t(); }

      HMP_HOST_DEVICE static constexpr char32_t
      denorm_min() noexcept { return char32_t(); }

      static constexpr bool is_iec559 = false;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = !is_signed;

      static constexpr bool traps = __glibcxx_integral_traps;
      static constexpr bool tinyness_before = false;
      static constexpr float_round_style round_style = round_toward_zero;
    };
#endif

  /// numeric_limits<short> specialization.
  template<>
    struct numeric_limits<short>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr short
      min() noexcept { return -__SHRT_MAX__ - 1; }

      HMP_HOST_DEVICE static constexpr short
      max() noexcept { return __SHRT_MAX__; }

      HMP_HOST_DEVICE static constexpr short
      lowest() noexcept { return min(); }


      static constexpr int digits = __glibcxx_digits (short);
      static constexpr int digits10 = __glibcxx_digits10 (short);
      static constexpr int max_digits10 = 0;

      static constexpr bool is_signed = true;
      static constexpr bool is_integer = true;
      static constexpr bool is_exact = true;
      static constexpr int radix = 2;

      HMP_HOST_DEVICE static constexpr short
      epsilon() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr short
      round_error() noexcept { return 0; }

      static constexpr int min_exponent = 0;
      static constexpr int min_exponent10 = 0;
      static constexpr int max_exponent = 0;
      static constexpr int max_exponent10 = 0;

      static constexpr bool has_infinity = false;
      static constexpr bool has_quiet_NaN = false;
      static constexpr bool has_signaling_NaN = false;
      static constexpr float_denorm_style has_denorm
       = denorm_absent;
      static constexpr bool has_denorm_loss = false;

      HMP_HOST_DEVICE static constexpr short
      infinity() noexcept { return short(); }

      HMP_HOST_DEVICE static constexpr short
      quiet_NaN() noexcept { return short(); }

      HMP_HOST_DEVICE static constexpr short
      signaling_NaN() noexcept { return short(); }

      HMP_HOST_DEVICE static constexpr short
      denorm_min() noexcept { return short(); }

      static constexpr bool is_iec559 = false;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = false;

      static constexpr bool traps = __glibcxx_integral_traps;
      static constexpr bool tinyness_before = false;
      static constexpr float_round_style round_style
       = round_toward_zero;
    };

  /// numeric_limits<unsigned short> specialization.
  template<>
    struct numeric_limits<unsigned short>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr unsigned short
      min() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr unsigned short
      max() noexcept { return __SHRT_MAX__ * 2U + 1; }

      HMP_HOST_DEVICE static constexpr unsigned short
      lowest() noexcept { return min(); }


      static constexpr int digits
       = __glibcxx_digits (unsigned short);
      static constexpr int digits10
       = __glibcxx_digits10 (unsigned short);
      static constexpr int max_digits10 = 0;

      static constexpr bool is_signed = false;
      static constexpr bool is_integer = true;
      static constexpr bool is_exact = true;
      static constexpr int radix = 2;

      HMP_HOST_DEVICE static constexpr unsigned short
      epsilon() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr unsigned short
      round_error() noexcept { return 0; }

      static constexpr int min_exponent = 0;
      static constexpr int min_exponent10 = 0;
      static constexpr int max_exponent = 0;
      static constexpr int max_exponent10 = 0;

      static constexpr bool has_infinity = false;
      static constexpr bool has_quiet_NaN = false;
      static constexpr bool has_signaling_NaN = false;
      static constexpr float_denorm_style has_denorm
       = denorm_absent;
      static constexpr bool has_denorm_loss = false;

      HMP_HOST_DEVICE static constexpr unsigned short
      infinity() noexcept
      { return static_cast<unsigned short>(0); }

      HMP_HOST_DEVICE static constexpr unsigned short
      quiet_NaN() noexcept
      { return static_cast<unsigned short>(0); }

      HMP_HOST_DEVICE static constexpr unsigned short
      signaling_NaN() noexcept
      { return static_cast<unsigned short>(0); }

      HMP_HOST_DEVICE static constexpr unsigned short
      denorm_min() noexcept
      { return static_cast<unsigned short>(0); }

      static constexpr bool is_iec559 = false;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = true;

      static constexpr bool traps = __glibcxx_integral_traps;
      static constexpr bool tinyness_before = false;
      static constexpr float_round_style round_style
       = round_toward_zero;
    };

  /// numeric_limits<int> specialization.
  template<>
    struct numeric_limits<int>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr int
      min() noexcept { return -__INT_MAX__ - 1; }

      HMP_HOST_DEVICE static constexpr int
      max() noexcept { return __INT_MAX__; }

      HMP_HOST_DEVICE static constexpr int
      lowest() noexcept { return min(); }


      static constexpr int digits = __glibcxx_digits (int);
      static constexpr int digits10 = __glibcxx_digits10 (int);
      static constexpr int max_digits10 = 0;

      static constexpr bool is_signed = true;
      static constexpr bool is_integer = true;
      static constexpr bool is_exact = true;
      static constexpr int radix = 2;

      HMP_HOST_DEVICE static constexpr int
      epsilon() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr int
      round_error() noexcept { return 0; }

      static constexpr int min_exponent = 0;
      static constexpr int min_exponent10 = 0;
      static constexpr int max_exponent = 0;
      static constexpr int max_exponent10 = 0;

      static constexpr bool has_infinity = false;
      static constexpr bool has_quiet_NaN = false;
      static constexpr bool has_signaling_NaN = false;
      static constexpr float_denorm_style has_denorm
       = denorm_absent;
      static constexpr bool has_denorm_loss = false;

      HMP_HOST_DEVICE static constexpr int
      infinity() noexcept { return static_cast<int>(0); }

      HMP_HOST_DEVICE static constexpr int
      quiet_NaN() noexcept { return static_cast<int>(0); }

      HMP_HOST_DEVICE static constexpr int
      signaling_NaN() noexcept { return static_cast<int>(0); }

      HMP_HOST_DEVICE static constexpr int
      denorm_min() noexcept { return static_cast<int>(0); }

      static constexpr bool is_iec559 = false;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = false;

      static constexpr bool traps = __glibcxx_integral_traps;
      static constexpr bool tinyness_before = false;
      static constexpr float_round_style round_style
       = round_toward_zero;
    };

  /// numeric_limits<unsigned int> specialization.
  template<>
    struct numeric_limits<unsigned int>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr unsigned int
      min() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr unsigned int
      max() noexcept { return __INT_MAX__ * 2U + 1; }

      HMP_HOST_DEVICE static constexpr unsigned int
      lowest() noexcept { return min(); }


      static constexpr int digits
       = __glibcxx_digits (unsigned int);
      static constexpr int digits10
       = __glibcxx_digits10 (unsigned int);
      static constexpr int max_digits10 = 0;

      static constexpr bool is_signed = false;
      static constexpr bool is_integer = true;
      static constexpr bool is_exact = true;
      static constexpr int radix = 2;

      HMP_HOST_DEVICE static constexpr unsigned int
      epsilon() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr unsigned int
      round_error() noexcept { return 0; }

      static constexpr int min_exponent = 0;
      static constexpr int min_exponent10 = 0;
      static constexpr int max_exponent = 0;
      static constexpr int max_exponent10 = 0;

      static constexpr bool has_infinity = false;
      static constexpr bool has_quiet_NaN = false;
      static constexpr bool has_signaling_NaN = false;
      static constexpr float_denorm_style has_denorm
       = denorm_absent;
      static constexpr bool has_denorm_loss = false;

      HMP_HOST_DEVICE static constexpr unsigned int
      infinity() noexcept { return static_cast<unsigned int>(0); }

      HMP_HOST_DEVICE static constexpr unsigned int
      quiet_NaN() noexcept
      { return static_cast<unsigned int>(0); }

      HMP_HOST_DEVICE static constexpr unsigned int
      signaling_NaN() noexcept
      { return static_cast<unsigned int>(0); }

      HMP_HOST_DEVICE static constexpr unsigned int
      denorm_min() noexcept
      { return static_cast<unsigned int>(0); }

      static constexpr bool is_iec559 = false;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = true;

      static constexpr bool traps = __glibcxx_integral_traps;
      static constexpr bool tinyness_before = false;
      static constexpr float_round_style round_style
       = round_toward_zero;
    };

  /// numeric_limits<long> specialization.
  template<>
    struct numeric_limits<long>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr long
      min() noexcept { return -__LONG_MAX__ - 1; }

      HMP_HOST_DEVICE static constexpr long
      max() noexcept { return __LONG_MAX__; }

      HMP_HOST_DEVICE static constexpr long
      lowest() noexcept { return min(); }


      static constexpr int digits = __glibcxx_digits (long);
      static constexpr int digits10 = __glibcxx_digits10 (long);
      static constexpr int max_digits10 = 0;

      static constexpr bool is_signed = true;
      static constexpr bool is_integer = true;
      static constexpr bool is_exact = true;
      static constexpr int radix = 2;

      HMP_HOST_DEVICE static constexpr long
      epsilon() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr long
      round_error() noexcept { return 0; }

      static constexpr int min_exponent = 0;
      static constexpr int min_exponent10 = 0;
      static constexpr int max_exponent = 0;
      static constexpr int max_exponent10 = 0;

      static constexpr bool has_infinity = false;
      static constexpr bool has_quiet_NaN = false;
      static constexpr bool has_signaling_NaN = false;
      static constexpr float_denorm_style has_denorm
       = denorm_absent;
      static constexpr bool has_denorm_loss = false;

      HMP_HOST_DEVICE static constexpr long
      infinity() noexcept { return static_cast<long>(0); }

      HMP_HOST_DEVICE static constexpr long
      quiet_NaN() noexcept { return static_cast<long>(0); }

      HMP_HOST_DEVICE static constexpr long
      signaling_NaN() noexcept { return static_cast<long>(0); }

      HMP_HOST_DEVICE static constexpr long
      denorm_min() noexcept { return static_cast<long>(0); }

      static constexpr bool is_iec559 = false;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = false;

      static constexpr bool traps = __glibcxx_integral_traps;
      static constexpr bool tinyness_before = false;
      static constexpr float_round_style round_style
       = round_toward_zero;
    };

  /// numeric_limits<unsigned long> specialization.
  template<>
    struct numeric_limits<unsigned long>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr unsigned long
      min() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr unsigned long
      max() noexcept { return __LONG_MAX__ * 2UL + 1; }

      HMP_HOST_DEVICE static constexpr unsigned long
      lowest() noexcept { return min(); }


      static constexpr int digits
       = __glibcxx_digits (unsigned long);
      static constexpr int digits10
       = __glibcxx_digits10 (unsigned long);
      static constexpr int max_digits10 = 0;

      static constexpr bool is_signed = false;
      static constexpr bool is_integer = true;
      static constexpr bool is_exact = true;
      static constexpr int radix = 2;

      HMP_HOST_DEVICE static constexpr unsigned long
      epsilon() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr unsigned long
      round_error() noexcept { return 0; }

      static constexpr int min_exponent = 0;
      static constexpr int min_exponent10 = 0;
      static constexpr int max_exponent = 0;
      static constexpr int max_exponent10 = 0;

      static constexpr bool has_infinity = false;
      static constexpr bool has_quiet_NaN = false;
      static constexpr bool has_signaling_NaN = false;
      static constexpr float_denorm_style has_denorm
       = denorm_absent;
      static constexpr bool has_denorm_loss = false;

      HMP_HOST_DEVICE static constexpr unsigned long
      infinity() noexcept
      { return static_cast<unsigned long>(0); }

      HMP_HOST_DEVICE static constexpr unsigned long
      quiet_NaN() noexcept
      { return static_cast<unsigned long>(0); }

      HMP_HOST_DEVICE static constexpr unsigned long
      signaling_NaN() noexcept
      { return static_cast<unsigned long>(0); }

      HMP_HOST_DEVICE static constexpr unsigned long
      denorm_min() noexcept
      { return static_cast<unsigned long>(0); }

      static constexpr bool is_iec559 = false;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = true;

      static constexpr bool traps = __glibcxx_integral_traps;
      static constexpr bool tinyness_before = false;
      static constexpr float_round_style round_style
       = round_toward_zero;
    };

  /// numeric_limits<long long> specialization.
  template<>
    struct numeric_limits<long long>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr long long
      min() noexcept { return -__LONG_LONG_MAX__ - 1; }

      HMP_HOST_DEVICE static constexpr long long
      max() noexcept { return __LONG_LONG_MAX__; }

      HMP_HOST_DEVICE static constexpr long long
      lowest() noexcept { return min(); }


      static constexpr int digits
       = __glibcxx_digits (long long);
      static constexpr int digits10
       = __glibcxx_digits10 (long long);
      static constexpr int max_digits10 = 0;

      static constexpr bool is_signed = true;
      static constexpr bool is_integer = true;
      static constexpr bool is_exact = true;
      static constexpr int radix = 2;

      HMP_HOST_DEVICE static constexpr long long
      epsilon() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr long long
      round_error() noexcept { return 0; }

      static constexpr int min_exponent = 0;
      static constexpr int min_exponent10 = 0;
      static constexpr int max_exponent = 0;
      static constexpr int max_exponent10 = 0;

      static constexpr bool has_infinity = false;
      static constexpr bool has_quiet_NaN = false;
      static constexpr bool has_signaling_NaN = false;
      static constexpr float_denorm_style has_denorm
       = denorm_absent;
      static constexpr bool has_denorm_loss = false;

      HMP_HOST_DEVICE static constexpr long long
      infinity() noexcept { return static_cast<long long>(0); }

      HMP_HOST_DEVICE static constexpr long long
      quiet_NaN() noexcept { return static_cast<long long>(0); }

      HMP_HOST_DEVICE static constexpr long long
      signaling_NaN() noexcept
      { return static_cast<long long>(0); }

      HMP_HOST_DEVICE static constexpr long long
      denorm_min() noexcept { return static_cast<long long>(0); }

      static constexpr bool is_iec559 = false;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = false;

      static constexpr bool traps = __glibcxx_integral_traps;
      static constexpr bool tinyness_before = false;
      static constexpr float_round_style round_style
       = round_toward_zero;
    };

  /// numeric_limits<unsigned long long> specialization.
  template<>
    struct numeric_limits<unsigned long long>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr unsigned long long
      min() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr unsigned long long
      max() noexcept { return __LONG_LONG_MAX__ * 2ULL + 1; }

      HMP_HOST_DEVICE static constexpr unsigned long long
      lowest() noexcept { return min(); }


      static constexpr int digits
       = __glibcxx_digits (unsigned long long);
      static constexpr int digits10
       = __glibcxx_digits10 (unsigned long long);
      static constexpr int max_digits10 = 0;

      static constexpr bool is_signed = false;
      static constexpr bool is_integer = true;
      static constexpr bool is_exact = true;
      static constexpr int radix = 2;

      HMP_HOST_DEVICE static constexpr unsigned long long
      epsilon() noexcept { return 0; }

      HMP_HOST_DEVICE static constexpr unsigned long long
      round_error() noexcept { return 0; }

      static constexpr int min_exponent = 0;
      static constexpr int min_exponent10 = 0;
      static constexpr int max_exponent = 0;
      static constexpr int max_exponent10 = 0;

      static constexpr bool has_infinity = false;
      static constexpr bool has_quiet_NaN = false;
      static constexpr bool has_signaling_NaN = false;
      static constexpr float_denorm_style has_denorm
       = denorm_absent;
      static constexpr bool has_denorm_loss = false;

      HMP_HOST_DEVICE static constexpr unsigned long long
      infinity() noexcept
      { return static_cast<unsigned long long>(0); }

      HMP_HOST_DEVICE static constexpr unsigned long long
      quiet_NaN() noexcept
      { return static_cast<unsigned long long>(0); }

      HMP_HOST_DEVICE static constexpr unsigned long long
      signaling_NaN() noexcept
      { return static_cast<unsigned long long>(0); }

      HMP_HOST_DEVICE static constexpr unsigned long long
      denorm_min() noexcept
      { return static_cast<unsigned long long>(0); }

      static constexpr bool is_iec559 = false;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = true;

      static constexpr bool traps = __glibcxx_integral_traps;
      static constexpr bool tinyness_before = false;
      static constexpr float_round_style round_style
       = round_toward_zero;
    };



  /// numeric_limits<float> specialization.
  template<>
    struct numeric_limits<float>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr float
      min() noexcept { return __FLT_MIN__; }

      HMP_HOST_DEVICE static constexpr float
      max() noexcept { return __FLT_MAX__; }

      HMP_HOST_DEVICE static constexpr float
      lowest() noexcept { return -__FLT_MAX__; }


      static constexpr int digits = __FLT_MANT_DIG__;
      static constexpr int digits10 = __FLT_DIG__;
      static constexpr int max_digits10
	 = __glibcxx_max_digits10 (__FLT_MANT_DIG__);

      static constexpr bool is_signed = true;
      static constexpr bool is_integer = false;
      static constexpr bool is_exact = false;
      static constexpr int radix = __FLT_RADIX__;

      HMP_HOST_DEVICE static constexpr float
      epsilon() noexcept { return __FLT_EPSILON__; }

      HMP_HOST_DEVICE static constexpr float
      round_error() noexcept { return 0.5F; }

      static constexpr int min_exponent = __FLT_MIN_EXP__;
      static constexpr int min_exponent10 = __FLT_MIN_10_EXP__;
      static constexpr int max_exponent = __FLT_MAX_EXP__;
      static constexpr int max_exponent10 = __FLT_MAX_10_EXP__;

      static constexpr bool has_infinity = __FLT_HAS_INFINITY__;
      static constexpr bool has_quiet_NaN = __FLT_HAS_QUIET_NAN__;
      static constexpr bool has_signaling_NaN = has_quiet_NaN;
      static constexpr float_denorm_style has_denorm
	= bool(__FLT_HAS_DENORM__) ? denorm_present : denorm_absent;
      static constexpr bool has_denorm_loss
       = __glibcxx_float_has_denorm_loss;

      HMP_HOST_DEVICE static constexpr float
      infinity() noexcept { return __builtin_huge_valf(); }

      HMP_HOST_DEVICE static constexpr float
      quiet_NaN() noexcept { return __builtin_nanf(""); }

      HMP_HOST_DEVICE static constexpr float
      signaling_NaN() noexcept { return __builtin_nansf(""); }

      HMP_HOST_DEVICE static constexpr float
      denorm_min() noexcept { return __FLT_DENORM_MIN__; }

      static constexpr bool is_iec559
	= has_infinity && has_quiet_NaN && has_denorm == denorm_present;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = false;

      static constexpr bool traps = __glibcxx_float_traps;
      static constexpr bool tinyness_before
       = __glibcxx_float_tinyness_before;
      static constexpr float_round_style round_style
       = round_to_nearest;
    };

#undef __glibcxx_float_has_denorm_loss
#undef __glibcxx_float_traps
#undef __glibcxx_float_tinyness_before

  /// numeric_limits<double> specialization.
  template<>
    struct numeric_limits<double>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr double
      min() noexcept { return __DBL_MIN__; }

      HMP_HOST_DEVICE static constexpr double
      max() noexcept { return __DBL_MAX__; }

      HMP_HOST_DEVICE static constexpr double
      lowest() noexcept { return -__DBL_MAX__; }


      static constexpr int digits = __DBL_MANT_DIG__;
      static constexpr int digits10 = __DBL_DIG__;
      static constexpr int max_digits10
	 = __glibcxx_max_digits10 (__DBL_MANT_DIG__);

      static constexpr bool is_signed = true;
      static constexpr bool is_integer = false;
      static constexpr bool is_exact = false;
      static constexpr int radix = __FLT_RADIX__;

      HMP_HOST_DEVICE static constexpr double
      epsilon() noexcept { return __DBL_EPSILON__; }

      HMP_HOST_DEVICE static constexpr double
      round_error() noexcept { return 0.5; }

      static constexpr int min_exponent = __DBL_MIN_EXP__;
      static constexpr int min_exponent10 = __DBL_MIN_10_EXP__;
      static constexpr int max_exponent = __DBL_MAX_EXP__;
      static constexpr int max_exponent10 = __DBL_MAX_10_EXP__;

      static constexpr bool has_infinity = __DBL_HAS_INFINITY__;
      static constexpr bool has_quiet_NaN = __DBL_HAS_QUIET_NAN__;
      static constexpr bool has_signaling_NaN = has_quiet_NaN;
      static constexpr float_denorm_style has_denorm
	= bool(__DBL_HAS_DENORM__) ? denorm_present : denorm_absent;
      static constexpr bool has_denorm_loss
        = __glibcxx_double_has_denorm_loss;

      HMP_HOST_DEVICE static constexpr double
      infinity() noexcept { return __builtin_huge_val(); }

      HMP_HOST_DEVICE static constexpr double
      quiet_NaN() noexcept { return __builtin_nan(""); }

      HMP_HOST_DEVICE static constexpr double
      signaling_NaN() noexcept { return __builtin_nans(""); }

      HMP_HOST_DEVICE static constexpr double
      denorm_min() noexcept { return __DBL_DENORM_MIN__; }

      static constexpr bool is_iec559
	= has_infinity && has_quiet_NaN && has_denorm == denorm_present;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = false;

      static constexpr bool traps = __glibcxx_double_traps;
      static constexpr bool tinyness_before
       = __glibcxx_double_tinyness_before;
      static constexpr float_round_style round_style
       = round_to_nearest;
    };

#undef __glibcxx_double_has_denorm_loss
#undef __glibcxx_double_traps
#undef __glibcxx_double_tinyness_before

  /// numeric_limits<long double> specialization.
  template<>
    struct numeric_limits<long double>
    {
      static constexpr bool is_specialized = true;

      HMP_HOST_DEVICE static constexpr long double
      min() noexcept { return __LDBL_MIN__; }

      HMP_HOST_DEVICE static constexpr long double
      max() noexcept { return __LDBL_MAX__; }

      HMP_HOST_DEVICE static constexpr long double
      lowest() noexcept { return -__LDBL_MAX__; }


      static constexpr int digits = __LDBL_MANT_DIG__;
      static constexpr int digits10 = __LDBL_DIG__;
      static constexpr int max_digits10
	 = __glibcxx_max_digits10 (__LDBL_MANT_DIG__);

      static constexpr bool is_signed = true;
      static constexpr bool is_integer = false;
      static constexpr bool is_exact = false;
      static constexpr int radix = __FLT_RADIX__;

      HMP_HOST_DEVICE static constexpr long double
      epsilon() noexcept { return __LDBL_EPSILON__; }

      HMP_HOST_DEVICE static constexpr long double
      round_error() noexcept { return 0.5L; }

      static constexpr int min_exponent = __LDBL_MIN_EXP__;
      static constexpr int min_exponent10 = __LDBL_MIN_10_EXP__;
      static constexpr int max_exponent = __LDBL_MAX_EXP__;
      static constexpr int max_exponent10 = __LDBL_MAX_10_EXP__;

      static constexpr bool has_infinity = __LDBL_HAS_INFINITY__;
      static constexpr bool has_quiet_NaN = __LDBL_HAS_QUIET_NAN__;
      static constexpr bool has_signaling_NaN = has_quiet_NaN;
      static constexpr float_denorm_style has_denorm
	= bool(__LDBL_HAS_DENORM__) ? denorm_present : denorm_absent;
      static constexpr bool has_denorm_loss
	= __glibcxx_long_double_has_denorm_loss;

      HMP_HOST_DEVICE static constexpr long double
      infinity() noexcept { return __builtin_huge_vall(); }

      HMP_HOST_DEVICE static constexpr long double
      quiet_NaN() noexcept { return __builtin_nanl(""); }

      HMP_HOST_DEVICE static constexpr long double
      signaling_NaN() noexcept { return __builtin_nansl(""); }

      HMP_HOST_DEVICE static constexpr long double
      denorm_min() noexcept { return __LDBL_DENORM_MIN__; }

      static constexpr bool is_iec559
	= has_infinity && has_quiet_NaN && has_denorm == denorm_present;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = false;

      static constexpr bool traps = __glibcxx_long_double_traps;
      static constexpr bool tinyness_before =
					 __glibcxx_long_double_tinyness_before;
      static constexpr float_round_style round_style =
						      round_to_nearest;
    };

#undef __glibcxx_long_double_has_denorm_loss
#undef __glibcxx_long_double_traps
#undef __glibcxx_long_double_tinyness_before

    template <>
    class numeric_limits<hmp::Half>
    {
    public:
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr bool has_infinity = true;
        static constexpr bool has_quiet_NaN = true;
        static constexpr bool has_signaling_NaN = true;
        static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
        static constexpr auto has_denorm_loss =
            numeric_limits<float>::has_denorm_loss;
        static constexpr auto round_style = numeric_limits<float>::round_style;
        static constexpr bool is_iec559 = true;
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo = false;
        static constexpr int digits = 11;
        static constexpr int digits10 = 3;
        static constexpr int max_digits10 = 5;
        static constexpr int radix = 2;
        static constexpr int min_exponent = -13;
        static constexpr int min_exponent10 = -4;
        static constexpr int max_exponent = 16;
        static constexpr int max_exponent10 = 4;
        static constexpr auto traps = numeric_limits<float>::traps;
        static constexpr auto tinyness_before =
            numeric_limits<float>::tinyness_before;
        HMP_HOST_DEVICE static constexpr hmp::Half min()
        {
            return hmp::Half(0x0400, hmp::Half::from_bits());
        }
        HMP_HOST_DEVICE static constexpr hmp::Half lowest()
        {
            return hmp::Half(0xFBFF, hmp::Half::from_bits());
        }
        HMP_HOST_DEVICE static constexpr hmp::Half max()
        {
            return hmp::Half(0x7BFF, hmp::Half::from_bits());
        }
        HMP_HOST_DEVICE static constexpr hmp::Half epsilon()
        {
            return hmp::Half(0x1400, hmp::Half::from_bits());
        }
        HMP_HOST_DEVICE static constexpr hmp::Half round_error()
        {
            return hmp::Half(0x3800, hmp::Half::from_bits());
        }
        HMP_HOST_DEVICE static constexpr hmp::Half infinity()
        {
            return hmp::Half(0x7C00, hmp::Half::from_bits());
        }
        HMP_HOST_DEVICE static constexpr hmp::Half quiet_NaN()
        {
            return hmp::Half(0x7E00, hmp::Half::from_bits());
        }
        HMP_HOST_DEVICE static constexpr hmp::Half signaling_NaN()
        {
            return hmp::Half(0x7D00, hmp::Half::from_bits());
        }
        HMP_HOST_DEVICE static constexpr hmp::Half denorm_min()
        {
            return hmp::Half(0x0001, hmp::Half::from_bits());
        }
    };

} // namespace hmp

#undef __glibcxx_signed
#undef __glibcxx_min
#undef __glibcxx_max
#undef __glibcxx_digits
#undef __glibcxx_digits10
#undef __glibcxx_max_digits10

#endif // ON_DEVICE_GLIBC_LIKE_NUMERIC_LIMITS_H_