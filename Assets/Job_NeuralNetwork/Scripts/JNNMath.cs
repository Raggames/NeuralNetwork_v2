using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Assets.Job_NeuralNetwork.Scripts.JNNFeedForwardLayer;

namespace Assets.Job_NeuralNetwork.Scripts
{
    public static class JNNMath
    {
        public enum ActivationFunctions
        {
            Linear,
            ReLU,
            PReLU,
            ELU,
            Sigmoid,
            Boolean,
            Softmax,
            Tanh,
            Sinusoid,
            Gaussian,
        }

        public static double ComputeActivation(ActivationFunctions type, bool derivative, double x, double[] a = null) 
        {
            double result = 0;
            switch (type)
            {
                case ActivationFunctions.Linear:
                    if (!derivative)
                    {
                        result = x;
                    }
                    else
                    {
                        result = x;
                    }

                    break;
                case ActivationFunctions.Sigmoid:
                    if (!derivative)
                    {
                        result = Logistic(x);
                    }
                    else
                    {
                        result = DLogistic(x);
                    }

                    break;
                case ActivationFunctions.ReLU:
                    if (!derivative)
                    {
                        result = ReLU(x);
                    }
                    else
                    {
                        result = DReLU(x);
                    }

                    break;
                case ActivationFunctions.Softmax:
                    if (!derivative)
                    {

                    }
                    else
                    {

                    }

                    break;
                case ActivationFunctions.Tanh:
                    if (!derivative)
                    {
                        result = Tanh(x);
                    }
                    else
                    {
                        result = DTanh(x);
                    }

                    break;
                case ActivationFunctions.Sinusoid:
                    if (!derivative)
                    {
                        result = Sinusoid(x);
                    }
                    else
                    {
                        result = DSinusoid(x);
                    }

                    break;
                case ActivationFunctions.Gaussian:
                    if (!derivative)
                    {
                        result = Gaussian(x);
                    }
                    else
                    {
                        result = DGaussian(x);
                    }

                    break;
                case ActivationFunctions.PReLU:
                    if (!derivative)
                    {
                        result = PReLU(x, a[0]);
                    }
                    else
                    {
                        result = DPReLU(x, a[0]);
                    }

                    break;
                case ActivationFunctions.ELU:
                    if (!derivative)
                    {
                        result = ELU(x, a[0]);
                    }
                    else
                    {
                        result = DELU(x, a[0]);
                    }

                    break;
            }
            return result;

        }

        public static double ReLU(double x)
        {
            return Math.Max(0, x);// x < 0 ? 0 : x;
        }

        public static double DReLU(double x)
        {
            return Math.Max(0, 1);// x < 0 ? 0 : x;
        }

        public static double Logistic(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));
        }

        public static double DLogistic(double x)
        {
            return Logistic(x) * (1 - Logistic(x));
        }

        public static double Tanh(double x)
        {
            return 2 / (1 + Math.Pow(Math.E, -(2 * x))) - 1;
        }

        public static double DTanh(double x)
        {
            return 1 - Math.Pow(Tanh(x), 2);
        }

        public static double Sinusoid(double x)
        {
            return Math.Sin(x);
        }

        public static double DSinusoid(double x)
        {
            return Math.Cos(x);
        }

        public static double Gaussian(double x)
        {
            return Math.Pow(Math.E, Math.Pow(-x, 2));
        }

        public static double DGaussian(double x)
        {
            return -2 * x * Math.Pow(Math.E, Math.Pow(-x, 2));
        }

        public static double PReLU(double x, double a)
        {
            return x < 0 ? a * x : x;
        }

        public static double DPReLU(double x, double a)
        {
            return x < 0 ? a : 1;
        }

        //Exponential Linear Unit 
        public static double ELU(double x, double a)
        {
            return x < 0 ? a * (Math.Pow(Math.E, x) - 1) : x;
        }

        public static double DELU(double x, double a)
        {
            return x < 0 ? ELU(x, a) + a : 1;
        }
    }
}
