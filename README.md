from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Complex:
    real: float
    imag: float
    
    def __str__(self) -> str:
        if self.imag >= 0:
            return f"{self.real} + {self.imag}i"
        else:
            return f"{self.real} - {abs(self.imag)}i"
    
    def add(self, other: Complex) -> Complex:
        match other:
            case Complex(real, imag):
                return Complex(self.real + real, self.imag + imag)
    
    def subtract(self, other: Complex) -> Complex:
        match other:
            case Complex(real, imag):
                return Complex(self.real - real, self.imag - imag)
    
    def multiply(self, other: Complex) -> Complex:
        match other:
            case Complex(real, imag):
                new_real = self.real * real - self.imag * imag
                new_imag = self.real * imag + self.imag * real
                return Complex(new_real, new_imag)
    
    def conjugate(self) -> Complex:
        return Complex(self.real, -self.imag)
    
    def modulus(self) -> float:
        return math.sqrt(self.real**2 + self.imag**2)

@dataclass(frozen=True)
class Vector2:
    x: float
    y: float
    
    def __str__(self) -> str:
        return f"({self.x}, {self.y})"
    
    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)
    
    def dot_product(self, other: Vector2) -> float:
        match other:
            case Vector2(x, y):
                return self.x * x + self.y * y
    
    def cross_product(self, other: Vector2) -> float:
        match other:
            case Vector2(x, y):
                return self.x * y - self.y * x
    
    def add(self, other: Vector2) -> Vector2:
        match other:
            case Vector2(x, y):
                return Vector2(self.x + x, self.y + y)
    
    def subtract(self, other: Vector2) -> Vector2:
        match other:
            case Vector2(x, y):
                return Vector2(self.x - x, self.y - y)
    
    def multiply_scalar(self, scalar: float) -> Vector2:
        return Vector2(self.x * scalar, self.y * scalar)

@dataclass(frozen=True)
class Matrix2:
    a11: float
    a12: float
    a21: float
    a22: float
    
    def __str__(self) -> str:
        return f"[{self.a11} {self.a12}]\n[{self.a21} {self.a22}]"
    
    def determinant(self) -> float:
        return self.a11 * self.a22 - self.a12 * self.a21
    
    def inverse(self) -> Matrix2 | None:
        det = self.determinant()
        if abs(det) < 1e-10:
            return None
        inv_det = 1.0 / det
        return Matrix2(
            self.a22 * inv_det, -self.a12 * inv_det,
            -self.a21 * inv_det, self.a11 * inv_det
        )
    
    def multiply(self, other: Matrix2) -> Matrix2:
        match other:
            case Matrix2(b11, b12, b21, b22):
                return Matrix2(
                    self.a11 * b11 + self.a12 * b21,
                    self.a11 * b12 + self.a12 * b22,
                    self.a21 * b11 + self.a22 * b21,
                    self.a21 * b12 + self.a22 * b22
                )
    
    def multiply_vector(self, vector: Vector2) -> Vector2:
        match vector:
            case Vector2(x, y):
                new_x = self.a11 * x + self.a12 * y
                new_y = self.a21 * x + self.a22 * y
                return Vector2(new_x, new_y)
    
    def transpose(self) -> Matrix2:
        return Matrix2(self.a11, self.a21, self.a12, self.a22)

def test_complex_numbers():
    print("=== COMPLEX NUMBERS ===")
    z1 = Complex(3, 4)
    z2 = Complex(1, 2)
    print(f"z1 = {z1}")
    print(f"z2 = {z2}")
    print(f"z1 + z2 = {z1.add(z2)}")
    print(f"z1 - z2 = {z1.subtract(z2)}")
    print(f"z1 * z2 = {z1.multiply(z2)}")
    print(f"conjugate z1 = {z1.conjugate()}")
    print(f"modulus z1 = {z1.modulus():.2f}")

def test_vectors():
    print("\n=== VECTORS 2D ===")
    v1 = Vector2(3, 4)
    v2 = Vector2(1, 2)
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"norm v1 = {v1.norm():.2f}")
    print(f"dot product = {v1.dot_product(v2)}")
    print(f"cross product = {v1.cross_product(v2)}")
    print(f"v1 + v2 = {v1.add(v2)}")
    print(f"v1 - v2 = {v1.subtract(v2)}")

def test_matrices():
    print("\n=== MATRICES 2x2 ===")
    m1 = Matrix2(1, 2, 3, 4)
    m2 = Matrix2(2, 0, 1, 2)
    v = Vector2(1, 1)
    print("Matrix m1:")
    print(m1)
    print(f"determinant m1 = {m1.determinant()}")
    print("\nMatrix m2:")
    print(m2)
    print("\nProduct m1 * m2:")
    product = m1.multiply(m2)
    print(product)
    print("\nInverse of m1:")
    inv = m1.inverse()
    if inv:
        print(inv)
    else:
        print("Matrix is singular")
    print(f"\nMultiply m1 by vector {v}:")
    result = m1.multiply_vector(v)
    print(result)

if __name__ == "__main__":
    test_complex_numbers()
    test_vectors()
    test_matrices()
