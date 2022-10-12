


class Matrix():
    def __init__(self, values=None):
        if (values is None):
            raise ValueError("No arugment provided")
        else:
            if (isinstance(values, list)):
                if (not all(isinstance(item, list) and all(isinstance(el, float) for el in item) for item in values)):
                    raise ValueError("Values invalid format !")
                self.data = values
                self.shape = (len(values), len(values[0]))
            elif (isinstance(values, tuple)):
                if (not all(isinstance(item, int) and item >= 0 for item in values) and len(values) != 2):
                    raise ValueError("Invalid shape format/type")
                self.data = [[float(i)] for i in range(values[0], values[1], 1)]
                self.shape = values
            else:
                raise ValueError("Invalid argument type.")

    def T(self):
        matrix = [[] for _ in range(self.shape[1])]
        for x, el in enumerate(matrix):
            for y in range(self.shape[0]):
                el.append(self.data[y][x])
        return Matrix(matrix)

    def __add__(self, matrix):
        if (isinstance(matrix, Matrix)):
            if (self.shape == matrix.shape):
                    return Matrix([[float(k) + float(l) for (k, l) in zip(i, j)] for (i, j) in zip(self.data, matrix.data)])
            else:
                raise ValueError("Matrixs does not have same shape !")
        raise TypeError("Can only add matrix to matrix")

    def __radd__(self, matrix):
        if (isinstance(matrix, Matrix)):
            if (self.shape == matrix.shape):
                    return Matrix([[float(k) + float(l) for (k, l) in zip(i, j)] for (i, j) in zip(self.data, matrix.data)])
            else:
                raise ValueError("Matrixs does not have same shape !")
        raise TypeError("Can only add matrix to matrix")

    def __sub__(self, matrix):
        if (isinstance(matrix, Matrix)):
            if (self.shape == matrix.shape):
                    return Matrix([[float(k) - float(l) for (k, l) in zip(i, j)] for (i, j) in zip(self.data, matrix.data)])
            else:
                raise ValueError("Matrixs does not have same shape !")
        raise TypeError("Can only add matrix to matrix")

    def __rsub__(self, matrix):
        if (isinstance(matrix, Matrix)):
            if (self.shape == matrix.shape):
                    return Matrix([[float(k) - float(l) for (k, l) in zip(i, j)] for (i, j) in zip(self.data, matrix.data)])
            else:
                raise ValueError("Matrixs does not have same shape !")
        raise TypeError("Can only add matrix to matrix")

    def __truediv__(self, scalar):
        if (isinstance(scalar, int) or isinstance(scalar, float)):
            if (scalar == 0):
                raise ZeroDivisionError()
            return Matrix([[float(k) / float(scalar) for k in i] for i in self.data])
        else:
            raise TypeError("Matrix can only be divided by int or float.")

    def __rtruediv__(self, scalar):
        if (isinstance(scalar, int) or isinstance(scalar, float)):
            if (scalar == 0):
                raise ZeroDivisionError()
            return Matrix([[float(scalar) / float(k) for k in i] for i in self.data])
        else:
            raise TypeError("Matrix can only be divided by int or float.")

    def __mul__(self, scalar):
        if (isinstance(scalar, int) or isinstance(scalar, float)):
            return Matrix([[float(scalar) * float(k) for k in i] for i in self.data])
        elif (isinstance(scalar, Matrix)):
            scalar_col = [[[row[col]] for row in scalar.data] for col in range(scalar.shape[1])]
            return Matrix([[Vector([[i] for i in row_data]).dot(Vector(col)) for col in scalar_col] for row_data in self.data])
        elif (isinstance(scalar, Vector)):
            print("here")
            return Matrix([[Vector([[i] for i in row_data]).dot(scalar)] for row_data in self.data])
        else:
            raise TypeError("Matrix can only be multiply by int/float, Matrix or Vector")

    def __rmul__(self, scalar):
        if (isinstance(scalar, int) or isinstance(scalar, float)):
            return Matrix([[float(scalar) * float(k) for k in i] for i in self.data])
        elif (isinstance(scalar, Matrix)):
            scalar_col = [[[row[col]] for row in scalar.data] for col in range(scalar.shape[1])]
            return Matrix([[Vector([[i] for i in row_data]).dot(Vector(col)) for col in scalar_col] for row_data in self.data])
        elif (isinstance(scalar, Vector)):
            return Matrix([[Vector([[i] for i in row_data]).dot(scalar)] for row_data in self.data])
        else:
            raise TypeError("Matrix can only be multiply by int/float, Matrix or Vector")

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

class Vector(Matrix):
    def __init__(self, values):
        if (not isinstance(values, list) or not all(isinstance(item, list) for item in values)):
            raise ValueError("Values invalid format !")
        if (len(values) == 1):
            self.shape = (1, len(values[0]))
        else:
            self.shape = (len(values), 1)
        if (len(values) == 0 or not all(len(lst) != 0 for lst in values)):
            raise ValueError("Some values are empty !")
        for lst in values:
            for value in lst:
                if (not (isinstance(value, float) or isinstance(value, int))):
                    raise ValueError("Some values are not float !")
        self.values = values

    def T(self):
        if (self.shape[0] == 1):
            return Vector([[float(i)] for i in self.values[0]])
        else:
            return Vector([[float(i[0]) for i in self.values]])

    def dot(self, vector):
        if (isinstance(vector, Vector)):
            if (self.shape == vector.shape):
                if (self.shape[0] == 1):
                    return sum([float(i) * float(j) for (i, j) in zip(self.values[0], vector.values[0])])
                else:
                    return sum([float(i[0]) * float(j[0]) for (i, j) in zip(self.values, vector.values)])
            else:
                raise ValueError("Vectors does not have same shape !")
        raise TypeError("Can only operate vector to vector")

    def __add__(self, vector):
        if (isinstance(vector, Vector)):
            if (self.shape == vector.shape):
                if (self.shape[0] == 1):
                    return Vector([[float(i) + float(j) for (i, j) in zip(self.values[0], vector.values[0])]])
                else:
                    return Vector([[float(i[0]) + float(j[0])] for (i, j) in zip(self.values, vector.values)])
            else:
                raise ValueError("Vectors does not have same shape !")
        raise TypeError("Can only add vector to vector")

    def __radd__(self, vector):
        if (isinstance(vector, Vector)):
            if (self.shape == vector.shape):
                if (self.shape[0] == 1):
                    return Vector([[float(i) + float(j) for (i, j) in zip(self.values[0], vector.values[0])]])
                else:
                    return Vector([[float(i[0]) + float(j[0])] for (i, j) in zip(self.values, vector.values)])
            else:
                raise ValueError("Vectors does not have same shape !")
        raise TypeError("Can only operate vector to vector")

    def __sub__(self, vector):
        if (isinstance(vector, Vector)):
            if (self.shape == vector.shape):
                if (self.shape[0] == 1):
                    return Vector([[float(i) - float(j) for (i, j) in zip(self.values[0], vector.values[0])]])
                else:
                    return Vector([[float(i[0]) - float(j[0])] for (i, j) in zip(self.values, vector.values)])
            else:
                raise ValueError("Vectors does not have same shape !")
        raise TypeError("Can only operate vector to vector")

    def __rsub__(self, vector):
        if (isinstance(vector, Vector)):
            if (self.shape == vector.shape):
                if (self.shape[0] == 1):
                    return Vector([[float(i) - float(j) for (i, j) in zip(self.values[0], vector.values[0])]])
                else:
                    return Vector([[float(i[0]) - float(j[0])] for (i, j) in zip(self.values, vector.values)])
            else:
                raise ValueError("Vectors does not have same shape !")
        raise TypeError("Can only operate vector to vector")

    def __truediv__(self, scalar):
        if (isinstance(scalar, int) or isinstance(scalar, float)):
            if (scalar == 0):
                raise ZeroDivisionError()
            if (self.shape[0] == 1):
                return Vector([[float(i) / scalar for i in self.values[0]]])
            else:
                return Vector([[float(i[0]) / scalar] for i in self.values])
        elif (isinstance(scalar, Vector)):
            raise NotImplementedError("Vector division not implemented.")
        else:
            raise TypeError("Vector can only be divided by int or float.")
        # truediv : only with scalars (to perform division of Vector by a scalar).

    def __rtruediv__(self, scalar):
        raise NotImplementedError("Division of a scalar by a Vector is not defined here.")

    def __mul__(self, scalar):
        if (isinstance(scalar, int) or isinstance(scalar, float)):
            if (self.shape[0] == 1):
                return Vector([[float(i) * scalar for i in self.values[0]]])
            else:
                return Vector([[float(i[0]) * scalar] for i in self.values])
        elif (isinstance(scalar, Vector)):
            raise NotImplementedError()
        elif (isinstance(scalar, Matrix)):
            return Matrix([[Vector([[i] for i in row_data]).dot(self)] for row_data in scalar.data])
        else:
            raise TypeError("Vector can only be multiply by int or float")
    def __rmul__(self, scalar):
        if (isinstance(scalar, int) or isinstance(scalar, float)):
            if (self.shape[0] == 1):
                return Vector([[float(i) * scalar for i in self.values[0]]])
            else:
                return Vector([[float(i[0]) * scalar] for i in self.values])
        elif (isinstance(scalar, Vector)):
            raise NotImplementedError()
        elif (isinstance(scalar, Matrix)):
            return Matrix([[Vector([[i] for i in row_data]).dot(self)] for row_data in scalar.data])
        else:
            raise TypeError("Vector can only be multiply by int or float")

    def __str__(self):
        return str(self.values)

    def __repr__(self):
        return str(self.values)

