using NUnit.Framework;
using UnityEngine;


[TestFixture]
public class MathTests 
{

    [TestCase(1, 2, 3, 4, 5, 6)]
    public void CheckMatrixMultiply(int m1, int m2, int m3, int m4, int v1, int v2)
    {
        var matrix = new double[,] { { m1, m2 }, { m3, m4 } };
        var vector = new double[] { v1, v2 };   


    }
}
