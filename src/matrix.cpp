#include <stdexcept>

#include "../include/matrix.hpp"


Matrix::Matrix(Matrix&& matrix) noexcept
{
        row_ = matrix.row_;
        col_ = matrix.col_;
        data = std::move(matrix.data);
        matrix.col_ = 0;
        matrix.row_ = 0;
}

Matrix& Matrix::operator=(Matrix&& matrix) noexcept
{
    if (this != &matrix) {
        row_ = matrix.row_;
        col_ = matrix.col_;
        data = std::move(matrix.data);
        matrix.col_ = 0;
        matrix.row_ = 0;
    }
    return *this;
}

// matrix operation
Matrix Matrix::T() const
{
    Matrix resMat(col_, row_);
    for (size_t i = 0; i < resMat.row_; ++i) {
        for (size_t j = 0; j < resMat.col_; ++j) {
            resMat.data[i][j] = data[j][i];
        }
    }

    return resMat;
}

Matrix Matrix::add(const Matrix& matrix) const
{
    if (row_ != matrix.row_ || col_ != matrix.col_) {
        throw std::invalid_argument("Matrix dimensions must match for add");
    }
    
    Matrix resMat(row_, col_);
    for (size_t i = 0; i < resMat.row_; ++i) {
        for (size_t j = 0; j < resMat.col_; ++j) {
            resMat.data[i][j] = data[i][j] + matrix.data[i][j];
        }
    }

    return resMat;
}
Matrix Matrix::sub(const Matrix& matrix) const
{
    if (row_ != matrix.row_ || col_ != matrix.col_) {
        throw std::invalid_argument("Matrix dimensions must match for sub");
    }
    
    Matrix resMat(row_, col_);
    for (size_t i = 0; i < resMat.row_; ++i) {
        for (size_t j = 0; j < resMat.col_; ++j) {
            resMat.data[i][j] = data[i][j] - matrix.data[i][j];
        }
    }

    return resMat;
}
Matrix Matrix::matMul(const Matrix& matrix) const
{
    if (col_ != matrix.row_) {
        throw std::invalid_argument("Matrix dimensions must match for Matrix-Mul");
    }

    Matrix resMat(row_, matrix.col_);
    for (size_t i = 0; i < resMat.row_; ++i) {
        // traverses rows continuously for better cache locality
        for (size_t k = 0; k < col_; k++) {
            float data_ik = data[i][k];
            for (size_t j = 0; j < resMat.col_; ++j) {
                resMat.data[i][j] += data_ik * matrix.data[k][j];
            }
        }
    }

    return resMat;
}
Matrix Matrix::elementMul(const Matrix& matrix) const
{
    if (row_ != matrix.row_ || col_ != matrix.col_) {
        throw std::invalid_argument("Matrix dimensions must match for element-Mul");
    }
    
    Matrix resMat(row_, col_);
    for (size_t i = 0; i < resMat.row_; ++i) {
        for (size_t j = 0; j < resMat.col_; ++j) {
            resMat.data[i][j] = data[i][j] * matrix.data[i][j];
        }
    }

    return resMat;
}

Matrix Matrix::add(float scalar) const
{
    Matrix resMat(*this);
    for (size_t i = 0; i < resMat.row_; ++i) {
        for (size_t j = 0; j < resMat.col_; ++j) {
            resMat.data[i][j] += scalar;
        }
    }

    return resMat;
}
Matrix Matrix::sub(float scalar) const
{
    Matrix resMat(*this);
    for (size_t i = 0; i < resMat.row_; ++i) {
        for (size_t j = 0; j < resMat.col_; ++j) {
            resMat.data[i][j] -= scalar;
        }
    }

    return resMat;
}
Matrix Matrix::elementMul(float scalar) const
{
    Matrix resMat(*this);
    for (size_t i = 0; i < resMat.row_; ++i) {
        for (size_t j = 0; j < resMat.col_; ++j) {
            resMat.data[i][j] *= scalar;
        }
    }

    return resMat;
}

Matrix Matrix::operator+(const Matrix& matrix) const
{
    return add(matrix);
}
Matrix Matrix::operator-(const Matrix& matrix) const
{
    return sub(matrix);
}
Matrix Matrix::operator*(const Matrix& matrix) const
{
    return matMul(matrix);
}


Matrix Matrix::operator+(float scalar) const
{
    return add(scalar);
}
Matrix Matrix::operator-(float scalar) const
{
    return sub(scalar);
}
Matrix Matrix::operator*(float scalar) const
{
    return elementMul(scalar);
}
Matrix Matrix::operator/(float scalar) const
{
    if (scalar == 0.0f) {
        throw std::invalid_argument("Division by zero in Matrix::operator/");
    }
    return elementMul(1.0f / scalar);
}

Matrix operator+(float scalar, const Matrix& matrix)
{
    return matrix.add(scalar);
}
Matrix operator*(float scalar, const Matrix& matrix)
{
    return matrix.elementMul(scalar);
}
