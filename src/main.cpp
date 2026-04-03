#include "../include/matrix.hpp"

#include <iostream>
#include <iomanip>


void printMatrix(const Matrix& mat, const std::string& name)
{
    std::cout << name << ":\n";
    for (size_t i = 0; i < mat.row(); ++i) {
        for (size_t j = 0; j < mat.col(); ++j) {
            std::cout << std::setw(8) << mat.get(i, j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}


int main()
{
    try {
        // 1. 构造和基础操作
        Matrix A(2, 3, 1.0f);   // 2x3，填充1.0
        Matrix B(2, 3, 2.0f);   // 2x3，填充2.0
        Matrix C(3, 2, 3.0f);   // 3x2，填充3.0

        std::cout << "=== Initial matrices ===\n";
        printMatrix(A, "A (2x3, fill=1)");
        printMatrix(B, "B (2x3, fill=2)");
        printMatrix(C, "C (3x2, fill=3)");

        // 2. 加法
        Matrix D = A.add(B);
        Matrix E = A + B;        // 运算符重载
        std::cout << "=== Addition ===\n";
        printMatrix(D, "A.add(B)");
        printMatrix(E, "A + B");

        // 3. 减法
        Matrix F = A.sub(B);
        Matrix G = A - B;
        std::cout << "=== Subtraction ===\n";
        printMatrix(F, "A.sub(B)");
        printMatrix(G, "A - B");

        // 4. 矩阵乘法
        Matrix H = A.matMul(C);
        Matrix I = A * C;
        std::cout << "=== Matrix multiplication (A * C) ===\n";
        printMatrix(H, "A.matMul(C)");
        printMatrix(I, "A * C");

        // 5. 逐元素乘法
        Matrix J = A.elementMul(B);
        std::cout << "=== Element-wise multiplication ===\n";
        printMatrix(J, "A.elementMul(B)");

        // 6. 转置
        Matrix AT = A.T();
        std::cout << "=== Transpose ===\n";
        printMatrix(AT, "A.T()");

        // 7. 标量运算
        Matrix K = A.add(5.0f);
        Matrix L = A * 2.0f;
        Matrix M = A / 2.0f;
        std::cout << "=== Scalar operations ===\n";
        printMatrix(K, "A.add(5)");
        printMatrix(L, "A * 2");
        printMatrix(M, "A / 2");

        // 8. 移动语义测试
        Matrix moved = std::move(A);
        std::cout << "=== After move ===\n";
        // A 现在处于空状态（row=0,col=0）
        printMatrix(moved, "moved from A");
        std::cout << "Original A now has size: " << A.row() << "x" << A.col() << "\n\n";

        // 9. 拷贝构造和赋值
        Matrix copy = B;
        Matrix assigned;
        assigned = B;
        std::cout << "=== Copy ===\n";
        printMatrix(copy, "copy of B");
        printMatrix(assigned, "assigned from B");

        // 10. 异常测试
        std::cout << "=== Exception tests ===\n";
        try {
            Matrix X(2,2,1);
            Matrix Y(3,3,1);
            Matrix Z = X.add(Y);  // 维度不匹配
        } catch (const std::invalid_argument& e) {
            std::cout << "Caught expected exception: " << e.what() << "\n";
        }

        try {
            Matrix X(2,2,1);
            Matrix Z = X / 0.0f;  // 除零
        } catch (const std::invalid_argument& e) {
            std::cout << "Caught division by zero: " << e.what() << "\n";
        }

        try {
            Matrix X(2,3,1);
            Matrix Y(4,5,1);
            Matrix Z = X.matMul(Y);  // 列数 != 行数
        } catch (const std::invalid_argument& e) {
            std::cout << "Caught matMul dimension mismatch: " << e.what() << "\n";
        }

        std::cout << "\nAll tests completed successfully.\n";

    } catch (const std::exception& e) {
        std::cerr << "Unexpected exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

