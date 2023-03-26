/*
 * Copyright 2023 Babit Authors
 *
 * This file is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This file is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 */

#ifndef CLOCK_MODULE_FRACTION_HPP
#define CLOCK_MODULE_FRACTION_HPP

#include <math.h>
#include <stdint.h>
#include <string>

namespace Fraction {

    class Fraction {
    public:
        Fraction() {
            neg_ = false;
            den_ = 0;
            base_ = 1;
        }

        Fraction(std::string const &frac) {
            neg_ = false;
            den_ = 0;
            base_ = 1;
            if (frac.empty()) {
                return;
            }
            int64_t a = -1, b = 1;
            size_t i = 0;
            std::string tmp = "";

            if (frac[0] == '-') {
                neg_ = true;
                ++i;
            }

            while (i < frac.length()) {
                if (frac[i] == '/') {
                    if (a >= 0)
                        throw std::logic_error("Wrong fraction provided.");
                    a = std::stol(tmp);
                    tmp = "";
                } else {
                    tmp += frac[i];
                }
                ++i;
            }
            if (!tmp.empty())
                b = std::stol(tmp);
            if (a < 0 || b <= 0)
                throw std::logic_error("Wrong fraction provided.");

            den_ = a, base_ = b;

            simplify();
        }

        Fraction(int32_t den, int32_t base) {
            neg_ = false;
            den_ = den, base_ = base;
            if (den < 0) {
                den_ = -den_;
                neg_ = !neg_;
            }
            if (base < 0) {
                base = -base_;
                neg_ = !neg_;
            }
            if (base == 0)
                throw std::logic_error("Wrong fraction provided.");

            simplify();
        }

        void simplify() {
            auto gcd_ = gcd(den_, base_);
            den_ /= gcd_, base_ /= gcd_;
            neg_ = den_ == 0 ? false : neg_;
            return;
        }

        bool operator==(Fraction const &rhs) {
            return neg_ == rhs.neg_ && den_ == rhs.den_ && base_ == rhs.base_;
        }

        bool operator>(Fraction const &rhs) {
            if (neg_ != rhs.neg_)
                return rhs.neg_;
            return !(*this - rhs).neg();
        }

        bool operator<(Fraction const &rhs) {
            if (neg_ != rhs.neg_)
                return neg_;
            return (*this - rhs).neg();
        }

        Fraction &operator*=(Fraction const &rhs) {
            neg_ = (neg_ != rhs.neg_);
            /*
           * a1/b1 * a2/b2
           * a1 gcd b1 == 1 && a2 gcd b2 == 1
           * (a1*a2) gcd (b1*b2) == (a1 gcd b2)*(a2 gcd b1)
           */
            auto den_gcd = gcd(rhs.den_, base_);
            auto base_gcd = gcd(rhs.base_, den_);
            den_ = (den_ / base_gcd) * (rhs.den_ / den_gcd);
            base_ = (base_ / den_gcd) * (rhs.base_ / base_gcd);

            return *this;
        }

        Fraction &operator*=(int32_t rhs) {
            if (rhs < 0) {
                rhs = -rhs;
                neg_ = !neg_;
            }
            auto gcd_ = gcd(base_, rhs);
            base_ /= gcd_;
            den_ *= (rhs / gcd_);

            return *this;
        }

        Fraction operator*(Fraction const &rhs) {
            Fraction res = *this;
            return res *= rhs;
        }

        Fraction operator*(int32_t rhs) {
            Fraction res = *this;
            return res *= rhs;
        }

        Fraction &operator/=(Fraction const &rhs) {
            neg_ = (neg_ != rhs.neg_);
            /*
             * (a1/b1) / (a2/b2) == (a1/b1) * (b2/a2);
             */
            auto den_gcd = gcd(rhs.den_, den_);
            auto base_gcd = gcd(rhs.base_, base_);
            den_ = (den_ / den_gcd) * (rhs.base_ / base_gcd);
            base_ = (base_ / base_gcd) * (rhs.den_ / den_gcd);

            return *this;
        }

        Fraction &operator/=(int32_t rhs) {
            if (rhs < 0) {
                rhs = -rhs;
                neg_ = !neg_;
            }
            auto gcd_ = gcd(den_, rhs);
            den_ /= gcd_;
            base_ *= (rhs / gcd_);

            return *this;
        }

        Fraction operator/(Fraction const &rhs) {
            Fraction res = *this;
            return res /= rhs;
        }

        Fraction operator/(int32_t rhs) {
            Fraction res = *this;
            return res /= rhs;
        }

        Fraction &operator+=(Fraction const &rhs) {
            /*
             * (a1/b1) + (a2/b2) == (a1*b2 + a2*b1)/(b1*b2)
             * Cannot be simplified by preprocess.
             */
            if (neg_ != rhs.neg_) {
                auto a = den_ * rhs.base_, b = rhs.den_ * base_;
                den_ = a > b ? a - b : b - a;
                if (a < b)
                    neg_ = !neg_;
            } else {
                den_ = den_ * rhs.base_ + rhs.den_ * base_;
            }

            base_ *= rhs.base_;

            simplify();
            return *this;
        }

        Fraction &operator+=(int32_t rhs) {
            /*
             * Cannot be simpler.
             */
            bool neg = rhs < 0;
            rhs = rhs < 0 ? -rhs : rhs;
            auto den = rhs * base_;
            if (neg_ != neg) {
                // Equal to {neg_ = den_ > den ? neg_ : !neg_;}
                neg_ = den_ > den == neg_;
                den_ = den_ > den ? den_ - den : den - den_;
            } else {
                den_ += den;
            }
            if (den_ == 0) {
                neg_ = false;
                base_ = 1;
            }
            return *this;
        }

        Fraction operator+(Fraction const &rhs) {
            Fraction res = *this;
            return res += rhs;
        }

        Fraction operator+(int32_t rhs) {
            Fraction res = *this;
            return res += rhs;
        }

        /*
         * +(a/b) = a/b
         * +(- a/b) = a/b
         */
        Fraction operator+() {
            Fraction res = *this;
            res.neg_ = false;
            return res;
        }

        Fraction &operator-=(Fraction const &rhs) {
            /*
             * a - b == a + (-b)
             */
            if (neg_ == rhs.neg_) {
                auto a = den_ * rhs.base_, b = rhs.den_ * base_;
                den_ = a > b ? a - b : b - a;
                if (a < b)
                    neg_ = !neg_;
            } else {
                den_ = den_ * rhs.base_ + rhs.den_ * base_;
            }

            base_ *= rhs.base_;

            simplify();
            return *this;
        }

        Fraction &operator-=(int32_t rhs) {
            /*
             * a - b == a + (-b)
             * Cannot be simpler.
             */
            bool neg = rhs < 0;
            rhs = rhs < 0 ? -rhs : rhs;
            auto den = rhs * base_;
            if (neg_ == neg) {
                // Equal to {neg_ = den_ > den ? neg_ : !neg_;}
                neg_ = den_ > den == neg_;
                den_ = den_ > den ? den_ - den : den - den_;
            } else {
                den_ += den;
            }
            if (den_ == 0) {
                neg_ = false;
                base_ = 1;
            }
            return *this;
        }

        Fraction operator-(Fraction const &rhs) {
            Fraction res = *this;
            return res -= rhs;
        }

        Fraction operator-(int32_t rhs) {
            Fraction res = *this;
            return res -= rhs;
        }

        /*
         * -(a/b) = (- a/b)
         * -(- a/b) = a/b
         */
        Fraction operator-() {
            Fraction res = *this;
            res.neg_ = !neg_;
            return res;
        }

        uint64_t &den() {
            return den_;
        }

        uint64_t &base() {
            return base_;
        }

        bool &neg() {
            return neg_;
        }

        Fraction to_based(Fraction const &base) {
            /*
             * Equal to (this/base).
             */
            auto res = *this;
            res /= base;
            return res;
        }

        int64_t to_int() {
            return neg_ ? -int64_t(den_ / base_) : int64_t(den_ / base_);
        }

        int64_t to_int_based(Fraction const &base) {
            /*
             * Equal to uint64_t(this/base).
             */
            auto res = *this;
            res /= base;
            return res.to_int();
        }

        double_t to_double() {
            return neg_ ? -(double_t(den_) / double_t(base_)) : double_t(den_) / double_t(base_);
        }

        double_t to_double_based(Fraction const &base) {
            /*
             * Equal to double_t(this/base).
             */
            auto res = *this;
            res /= base;
            return res.to_double();
        }

        uint64_t den_, base_;
        bool neg_;

    private:
        uint64_t gcd(uint64_t x, uint64_t y) {
            if (x < y)
                x ^= y, y ^= x, x ^= y;
            uint64_t t;
            while (y)
                t = y, y = x % y, x = t;
            return x;
        }
    };
}

#endif //CLOCK_MODULE_FRACTION_HPP
