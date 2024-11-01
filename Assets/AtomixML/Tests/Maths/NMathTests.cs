using Atom.MachineLearning.Core;
using NUnit.Framework;
using UnityEngine;


[TestFixture]
public class MathTests 
{
    [TestCase(1, 2, 3, 4, 5, 6, 17, 39)]
    [TestCase(2, 0, 7, 4, 2, 2, 4, 22)]
    public void Check_Matrix2x2_Multiply_Vector2(int m1, int m2, int m3, int m4, int v1, int v2, int vr1, int vr2)
    {
        var matrix = new NMatrix(new double[,] { { m1, m2 }, { m3, m4 } });
        var vector = new NVector(new double[] { v1, v2 });

        var check = new NVector(vr1, vr2);
        var result = matrix * vector;
        var result2 = vector * matrix;

        Assert.AreEqual(check.x, result.x);
        Assert.AreEqual(check.y, result.y);
    }

    [TestCase(1, -1, 2, 4, 3, 0, 7, 1, 2, 1, 2, 3, 5, 10, 15)]
    public void Check_Matrix3x3_Multiply_Vector3(int m1, int m2, int m3, int m4, int m5, int m6, int m7, int m8, int m9, int v1, int v2, int v3, int vr1, int vr2, int vr3)
    {
        var matrix = new NMatrix(new double[,] { { m1, m2 , m3 }, { m4, m5 , m6}, { m7, m8, m9 } });
        var vector = new NVector(new double[] { v1, v2, v3 });

        var check = new NVector(vr1, vr2, vr3);
        var result = matrix * vector;

        Assert.AreEqual(check.x, result.x);
        Assert.AreEqual(check.y, result.y);
        Assert.AreEqual(check.z, result.z);
    }

    [TestCase(0, 1, 2, 3)]
    public void Check_2_Vector2_DenseOfColumn_Matrix2x2(int v1, int v2, int v3, int v4)
    {
        var vector_a = new NVector(new double[] { v1, v2 });
        var vector_b = new NVector(new double[] { v3, v4 });

        var check = new NMatrix(new double[,] { { v1, v3 }, { v2, v4 } });
        var result = NMatrix.DenseOfColumnVectors(vector_a, vector_b);

        Assert.IsTrue(check == result);
    }
}
