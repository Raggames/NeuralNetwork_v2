using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Assets.Job_NeuralNetwork.Scripts
{
    public class JNNCPLayer : MonoBehaviour
    {
        public int matrixDim = 3;

        protected double[][] convertedMatrix;

        public Texture2D textureIn;
        public Texture2D textureOut;

    }
}
