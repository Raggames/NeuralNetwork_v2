using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Sirenix.OdinInspector;
using UnityEngine;

namespace Atom.MachineLearning.Core
{
    public class TestingScript : MonoBehaviour
    {
        [Button]
        private void TestMatrix3x3Determinant()
        {
            double[,] matrice = new double[,]
            {
            { 1f, 3f, 1f },
            { -1f, 7f, -9f },
            { 1f, -1f, 2f}
            };
            // = -22 
            var det = new NMatrix(matrice).LaplaceExpansionDeterminant();

            Debug.Log("Result : " + det);
        }

        [Button]
        private void TestMatrix4x4Determinant_1()
        {            
            var matrice = new double[,]
            {
            { 1f, 3f, 1f, -6f },
            { -1f, 7f, -9f, 17f },
            { 1f, -1f, 2f, -6f },
            { 1f, -1f, 3f, -6f },
            };

            var det = new NMatrix(matrice).LaplaceExpansionDeterminant();
            Debug.Log("Result : " + det);
        }

        [Button]
        private void TestMatrix4x4Determinant_2()
        {
            var matrice = new double[,]
            {
            { 1f, 2f, 3f, 4f },
            { 5f, 6f, -7f, 8f },
            { 9f, 10f, 11f, 12f },
            { 13f, 14f, 15f, 16f },
            };

            // 0    
            var det = new NMatrix(matrice).LaplaceExpansionDeterminant();
            Debug.Log("Result : " + det);
        }


        [Button]
        private NVector[] TestMatrixToNVectorArray()
        {
            var matrice = new double[,]
            {
            { 1f, 2f, 3f, 4f },
            { 5f, 6f, -7f, 8f },
            { 9f, 10f, 11f, 12f },
            { 13f, 14f, 15f, 16f },
            };

            var array = matrice.ToNVectorRowsArray();
            return array;
        }

        [Button]
        private double TestGaussian(double dist, double rad)
        {
            return MLMath.Gaussian(dist, rad);
        }

    }


}
