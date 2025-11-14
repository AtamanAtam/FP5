from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable, Tuple


@dataclass(frozen=True)
class Complex:
    real: float
    imag: float

def complex_to_str(z: Complex) -> str:
    if z.imag >= 0:
        return f"{z.real} + {z.imag}i"
    else:
        return f"{z.real} - {abs(z.imag)}i"

def complex_add(z1: Complex, z2: Complex) -> Complex:
    return Complex(z1.real + z2.real, z1.imag + z2.imag)

def complex_subtract(z1: Complex, z2: Complex) -> Complex:
    return Complex(z1.real - z2.real, z1.imag - z2.imag)

def complex_multiply(z1: Complex, z2: Complex) -> Complex:
    new_real = z1.real * z2.real - z1.imag * z2.imag
    new_imag = z1.real * z2.imag + z1.imag * z2.real
    return Complex(new_real, new_imag)

def complex_conjugate(z: Complex) -> Complex:
    return Complex(z.real, -z.imag)

def complex_modulus(z: Complex) -> float:
    return math.sqrt(z.real**2 + z.imag**2)


@dataclass(frozen=True)
class Vector2:
    x: float
    y: float

def vector2_to_str(v: Vector2) -> str:
    return f"({v.x}, {v.y})"

def vector2_norm(v: Vector2) -> float:
    return math.sqrt(v.x**2 + v.y**2)

def vector2_dot_product(v1: Vector2, v2: Vector2) -> float:
    return v1.x * v2.x + v1.y * v2.y

def vector2_cross_product(v1: Vector2, v2: Vector2) -> float:
    return v1.x * v2.y - v1.y * v2.x

def vector2_add(v1: Vector2, v2: Vector2) -> Vector2:
    return Vector2(v1.x + v2.x, v1.y + v2.y)

def vector2_subtract(v1: Vector2, v2: Vector2) -> Vector2:
    return Vector2(v1.x - v2.x, v1.y - v2.y)

def vector2_multiply_scalar(v: Vector2, scalar: float) -> Vector2:
    return Vector2(v.x * scalar, v.y * scalar)


@dataclass(frozen=True)
class Matrix2:
    a11: float
    a12: float
    a21: float
    a22: float

def matrix2_to_str(m: Matrix2) -> str:
    return f"[{m.a11} {m.a12}]\n[{m.a21} {m.a22}]"

def matrix2_determinant(m: Matrix2) -> float:
    return m.a11 * m.a22 - m.a12 * m.a21

def matrix2_inverse(m: Matrix2) -> Matrix2 | None:
    det = matrix2_determinant(m)
    if abs(det) < 1e-10:
        return None
    inv_det = 1.0 / det
    return Matrix2(
        m.a22 * inv_det, -m.a12 * inv_det,
        -m.a21 * inv_det, m.a11 * inv_det
    )

def matrix2_multiply(m1: Matrix2, m2: Matrix2) -> Matrix2:
    return Matrix2(
        m1.a11 * m2.a11 + m1.a12 * m2.a21,
        m1.a11 * m2.a12 + m1.a12 * m2.a22,
        m1.a21 * m2.a11 + m1.a22 * m2.a21,
        m1.a21 * m2.a12 + m1.a22 * m2.a22
    )

def matrix2_multiply_vector(m: Matrix2, v: Vector2) -> Vector2:
    new_x = m.a11 * v.x + m.a12 * v.y
    new_y = m.a21 * v.x + m.a22 * v.y
    return Vector2(new_x, new_y)

def matrix2_transpose(m: Matrix2) -> Matrix2:
    return Matrix2(m.a11, m.a21, m.a12, m.a22)


def test_complex_numbers():
    print("=== COMPLEX NUMBERS ===")
    z1 = Complex(3, 4)
    z2 = Complex(1, 2)
    print(f"z1 = {complex_to_str(z1)}")
    print(f"z2 = {complex_to_str(z2)}")
    print(f"z1 + z2 = {complex_to_str(complex_add(z1, z2))}")
    print(f"z1 - z2 = {complex_to_str(complex_subtract(z1, z2))}")
    print(f"z1 * z2 = {complex_to_str(complex_multiply(z1, z2))}")
    print(f"conjugate z1 = {complex_to_str(complex_conjugate(z1))}")
    print(f"modulus z1 = {complex_modulus(z1):.2f}")

def test_vectors():
    print("\n=== VECTORS 2D ===")
    v1 = Vector2(3, 4)
    v2 = Vector2(1, 2)
    print(f"v1 = {vector2_to_str(v1)}")
    print(f"v2 = {vector2_to_str(v2)}")
    print(f"norm v1 = {vector2_norm(v1):.2f}")
    print(f"dot product = {vector2_dot_product(v1, v2)}")
    print(f"cross product = {vector2_cross_product(v1, v2)}")
    print(f"v1 + v2 = {vector2_to_str(vector2_add(v1, v2))}")
    print(f"v1 - v2 = {vector2_to_str(vector2_subtract(v1, v2))}")

def test_matrices():
    print("\n=== MATRICES 2x2 ===")
    m1 = Matrix2(1, 2, 3, 4)
    m2 = Matrix2(2, 0, 1, 2)
    v = Vector2(1, 1)
    print("Matrix m1:")
    print(matrix2_to_str(m1))
    print(f"determinant m1 = {matrix2_determinant(m1)}")
    print("\nMatrix m2:")
    print(matrix2_to_str(m2))
    print("\nProduct m1 * m2:")
    product = matrix2_multiply(m1, m2)
    print(matrix2_to_str(product))
    print("\nInverse of m1:")
    inv = matrix2_inverse(m1)
    if inv:
        print(matrix2_to_str(inv))
    else:
        print("Matrix is singular")
    print(f"\nMultiply m1 by vector {vector2_to_str(v)}:")
    result = matrix2_multiply_vector(m1, v)
    print(vector2_to_str(result))

if __name__ == "__main__":
    test_complex_numbers()
    test_vectors()
    test_matrices()
