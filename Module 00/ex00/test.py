from matrix import Vector, Matrix
import unittest


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TestVector(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(ValueError):
            Vector("not list")  # not a list
        with self.assertRaises(ValueError):
            Vector([["lol"]])
        with self.assertRaises(ValueError):
            Vector([[]])
        with self.assertRaises(ValueError):
            Vector([])
        with self.assertRaises(ValueError):
            Vector(-5)
        with self.assertRaises(ValueError):
            Vector((2, 1))
        Vector([[5.], [8.], [9.], [10.]])
        Vector([[5., 8., 9., 10.]])
        print(bcolors.OKGREEN + "VECTOR INIT TEST SUCCESS" + bcolors.ENDC)

    def test_mult(self):
        v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
        v2 = v1 * 5
        self.assertEqual(v2.values, [[0.0], [5.0], [10.0], [15.0]])

        v1 = Vector([[0.0, 1.0, 2.0, 3.0]])
        v2 = v1 * 5
        self.assertEqual(v2.values, [[0.0, 5.0, 10.0, 15.0]])

        with self.assertRaises(NotImplementedError):
            v1 * v2
        with self.assertRaises(TypeError):
            v1 * "lol"
        print(bcolors.OKGREEN + "VECTOR MULT TEST SUCCESS" + bcolors.ENDC)

    def test_div(self):
        v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
        v2 = v1 * 5
        v2 = v2 / 5
        self.assertEqual(v2.values, v1.values)

        with self.assertRaises(ZeroDivisionError):
            v2 = v1 / 0
        with self.assertRaises(NotImplementedError):
            v1 / v2
        with self.assertRaises(TypeError):
            v1 / "lol"
        with self.assertRaises(NotImplementedError):
            5 / v1
        print(bcolors.OKGREEN + "VECTOR DIV TEST SUCCESS" + bcolors.ENDC)
        

    def test_add(self):
        v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
        v2 = Vector([[1.0], [2.0], [3.0], [4.0]])
        v2 = v1 + v2
        self.assertEqual(v2.values, [[1.0], [3.0], [5.0], [7.0]])

        v1 = Vector([[0.0, 1.0, 2.0, 3.0]])
        v2 = Vector([[1.0, 2.0, 3.0, 4.0]])
        v2 = v1 + v2
        self.assertEqual(v2.values, [[1.0, 3.0, 5.0, 7.0]])

        with self.assertRaises(ValueError):
            v2 = Vector([[1.0, 2.0, 3.0, 4.0, 5.0]])
            v1 = v1 + v2
        with self.assertRaises(TypeError):
            v1 = v1 + 5
        print(bcolors.OKGREEN + "VECTOR ADD TEST SUCCESS" + bcolors.ENDC)

    def test_sub(self):
        v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
        v2 = Vector([[1.0], [2.0], [3.0], [4.0]])
        v2 = v1 - v2
        self.assertEqual(v2.values, [[-1.0], [-1.0], [-1.0], [-1.0]])

        v1 = Vector([[0.0, 1.0, 2.0, 3.0]])
        v2 = Vector([[1.0, 2.0, 3.0, 4.0]])
        v2 = v1 - v2
        self.assertEqual(v2.values, [[-1.0, -1.0, -1.0, -1.0]])

        with self.assertRaises(ValueError):
            v2 = Vector([[1.0, 2.0, 3.0, 4.0, 5.0]])
            v1 = v1 - v2
        with self.assertRaises(TypeError):
            v1 = v1 - 5
        print(bcolors.OKGREEN + "VECTOR SUB TEST SUCCESS" + bcolors.ENDC)
    
    def test_T(self):
        v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
        self.assertEqual(v1.T().values, [[0.0, 1.0, 2.0, 3.0]])

        v1 = Vector([[0.0, 1.0, 2.0, 3.0]])
        self.assertEqual(v1.T().values, [[0.0], [1.0], [2.0], [3.0]])
        print(bcolors.OKGREEN + "VECTOR TRANSPOSE TEST SUCCESS" + bcolors.ENDC)

    def test_dot(self):
        v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
        v2 = Vector([[1.0], [2.0], [3.0], [4.0]])
        self.assertEqual(v1.dot(v2), 20)
        print(bcolors.OKGREEN + "VECTOR DOT TEST SUCCESS" + bcolors.ENDC)
    

class TestMatrix(unittest.TestCase):
    def test_T(self):
        m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        m2 = Matrix([[0., 2., 4.], [1., 3., 5.]])
        self.assertEqual(m1.T().data, m2.data)

        self.assertEqual(m2.T().data, m1.data)
        print(bcolors.OKGREEN + "MATRIX TRANSPOSE TEST SUCCESS" + bcolors.ENDC)
    
    def test_ADD(self):
        v1 = Vector([[0., 2., 4.]])
        m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        m2 = Matrix([[0., 2., 4.], [1., 3., 5.]])
        with self.assertRaises(ValueError):
            res = m1 + m2
        with self.assertRaises(TypeError):
            res = m1 + v1
        with self.assertRaises(TypeError):
            res = m1 + 5
        m2 = Matrix([[4.0, 9.0], [2.0, -3.0], [9.0, 0.0]])
        res = m1 + m2
        self.assertEqual(res.data, [[4.0, 10.0], [4.0, 0.0], [13.0, 5.0]])
        print(bcolors.OKGREEN + "MATRIX ADD TEST SUCCESS" + bcolors.ENDC)
    
    def test_SUB(self):
        v1 = Vector([[0., 2., 4.]])
        m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        m2 = Matrix([[0., 2., 4.], [1., 3., 5.]])
        with self.assertRaises(ValueError):
            res = m1 - m2
        with self.assertRaises(TypeError):
            res = m1 - v1
        with self.assertRaises(TypeError):
            res = m1 - 5
        m2 = Matrix([[4.0, 9.0], [2.0, -3.0], [9.0, 0.0]])
        res = m1 - m2
        self.assertEqual(res.data, [[-4.0, -8.0], [0.0, 6.0], [-5.0, 5.0]])
        print(bcolors.OKGREEN + "MATRIX SUB TEST SUCCESS" + bcolors.ENDC)
    
    def test_DIV(self):
        m1 = Matrix([[1.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        v1 = Vector([[0., 2., 4.]])
        m2 = Matrix([[0., 2., 4.], [1., 3., 5.]])
        with self.assertRaises(TypeError):
            res = m1 / m2
        with self.assertRaises(NotImplementedError):
            res = m1 / v1
        with self.assertRaises(ZeroDivisionError):
            res = m1 / 0
        m2 = Matrix([[4.0, 9.0], [2.0, -3.0], [9.0, 0.0]])
        res = m1 / 2
        self.assertEqual(res.data, [[0.5, 0.5], [1.0, 1.5], [2.0, 2.5]])
        res = 2 / m1
        self.assertEqual(res.data, [[2.0, 2.0], [1.0, 2/3], [0.5, 0.4]])
        print(bcolors.OKGREEN + "MATRIX SUB TEST SUCCESS" + bcolors.ENDC)

    def test_MULT(self):
        m1 = Matrix([[1.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        v1 = Vector([[0., 2., 4.]])
        m2 = Matrix([[0., 2., 4.], [1., 3., 5.]])
        res = m1 * m2
        self.assertEqual(res.data, [[1.0, 5.0, 9.0], [3.0, 13.0, 23.0], [5.0, 23.0, 41.0]])
        res = m2 * m1
        self.assertEqual(res.data, [[20.0, 26.0], [27.0, 35.0]])
        with self.assertRaises(ValueError):
            res = m2 * v1
        v1 = Vector([[0.], [2.], [4.]])
        with self.assertRaises(ValueError):
            res = m1 * v1
        res = m2 * v1
        self.assertEqual(res.data, [[20.0], [26.0]])
        res = v1 * m2
        self.assertEqual(res.data, [[20.0], [26.0]])
        print(bcolors.OKGREEN + "MATRIX MULT TEST SUCCESS" + bcolors.ENDC)

def main():
    m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    print(m1.shape)  # Output: (3, 2)
    print(m1.T())  # Output:  Matrix([[0., 2., 4.], [1., 3., 5.]])
    print(m1.T().shape) # Output  (2, 3)

    m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
    print(m1.shape)  # Output:  (2, 3)
    print(m1.T())  # Output:  Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    print(m1.T().shape)  # Output: (3, 2)

    m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
                 [0.0, 2.0, 4.0, 6.0]])
    m2 = Matrix([[0.0, 1.0],
                 [2.0, 3.0],
                 [4.0, 5.0],
                 [6.0, 7.0]])
    print(m1 * m2) # Output:  Matrix([[28., 34.], [56., 68.]])
    m1 = Matrix([[0.0, 1.0, 2.0],
                 [0.0, 2.0, 4.0]])
    v1 = Vector([[1], [2], [3]])
    print(m1 * v1)  # Output:  Matrix([[8], [16]]) Or: Vector([[8], [16])
    v1 = Vector([[1], [2], [3]])
    v2 = Vector([[2], [4], [8]])
    print(v1 + v2)  # Output:Vector([[3],[6],[11]])

if __name__ == '__main__':
    main()
    print("UNITTEST")
    unittest.main()
