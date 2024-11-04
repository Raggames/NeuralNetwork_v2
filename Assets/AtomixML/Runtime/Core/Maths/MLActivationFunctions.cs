using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Maths
{
    public static class MLActivationFunctions
    {
        public enum ActivationType
        {
            Identity,
            BinaryStep,
            Logistic,
            Tanh,
            ArcTan,
            ReLU,
            PReLU,
            ELU,
            SoftPlus,
            BentIdentity,
            Sinusoid,
            Sinc,
            Gaussian,
            Bipolar,
            BipolarSigmoid
        }

        public static double Activate(ActivationType type, double x)
        {
            switch (type)
            {
                case ActivationType.Identity:
                    return x;
                case ActivationType.BinaryStep:
                    return BinaryStep(x);
                case ActivationType.Logistic:
                    return Sigmoid(x);
                case ActivationType.Tanh:
                    return Tanh(x);
                case ActivationType.ArcTan:
                    return ArcTan(x);
                case ActivationType.ReLU:
                    return ReLU(x);
                case ActivationType.SoftPlus:
                    return SoftPlus(x);
                case ActivationType.BentIdentity:
                    return BentIdentity(x);
                case ActivationType.Sinusoid:
                    return Sinusoid(x);
                case ActivationType.Sinc:
                    return Sinc(x);
                case ActivationType.Gaussian:
                    return Gaussian(x);
                case ActivationType.Bipolar:
                    return Bipolar(x);
                case ActivationType.BipolarSigmoid:
                    return BipolarSigmoid(x);
            }
            return 0;
        }

        public static double Derivate(ActivationType activationType, double x)
        {
            switch (activationType)
            {
                case ActivationType.Logistic:
                    return DSigmoid(x);
                case ActivationType.Tanh:
                    return DTanh(x);
                case ActivationType.ArcTan:
                    return DArcTan(x);
                case ActivationType.ReLU:
                    return DReLU(x);
                case ActivationType.SoftPlus:
                    return DSoftPlus(x);
                case ActivationType.BentIdentity:
                    return DBentIdentity(x);
                case ActivationType.Sinusoid:
                    return DSinusoid(x);
                case ActivationType.Sinc:
                    return DSinc(x);
                case ActivationType.Gaussian:
                    return DGaussian(x);
                case ActivationType.BipolarSigmoid:
                    return DBipolarSigmoid(x);
            }
            return 0;
        }

        public static double BinaryStep(double x)
        {
            return x < 0 ? 0 : 1;
        }

        public static double Sigmoid(double x)
        {
            double result = 0;

            if (x > 45)
            {
                result = 1;
            }
            else
            {
                result = 1 / (1 + Math.Exp(-x));
            }
            return result;
        }

        public static double DSigmoid(double x)
        {
            return x * (1 - x);
        }

        public static double Tanh(double x)
        {
            return 2 / (1 + Math.Pow(Math.E, -(2 * x))) - 1;
        }
        public static double DTanh(double x)
        {
            return 1 - Math.Pow(Tanh(x), 2);
        }
        public static double ArcTan(double x)
        {
            return Math.Atan(x);
        }
        public static double DArcTan(double x)
        {
            return 1 / Math.Pow(x, 2) + 1;
        }

        //Rectified Linear Unit
        public static double ReLU(double x)
        {
            return Math.Max(0, x);// x < 0 ? 0 : x;
        }
        public static double DReLU(double x)
        {
            return Math.Max(0, 1);// x < 0 ? 0 : x;
        }

        //Parameteric Rectified Linear Unit 
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
        public static double SoftPlus(double x)
        {
            return Math.Log(Math.Exp(x) + 1);
        }
        public static double DSoftPlus(double x)
        {
            return Sigmoid(x);
        }
        public static double BentIdentity(double x)
        {
            return (((Math.Sqrt(Math.Pow(x, 2) + 1)) - 1) / 2) + x;
        }
        public static double DBentIdentity(double x)
        {
            return (x / (2 * Math.Sqrt(Math.Pow(x, 2) + 1))) + 1;
        }
        //  public float SoftExponential(float x)
        //  {
        //
        //  }
        public static double Sinusoid(double x)
        {
            return Math.Sin(x);
        }
        public static double DSinusoid(double x)
        {
            return Math.Cos(x);
        }
        public static double Sinc(double x)
        {
            return x == 0 ? 1 : Math.Sin(x) / x;
        }
        public static double DSinc(double x)
        {
            return x == 0 ? 0 : (Math.Cos(x) / x) - (Math.Sin(x) / Math.Pow(x, 2));
        }
        public static double Gaussian(double x)
        {
            return Math.Pow(Math.E, Math.Pow(-x, 2));
        }
        public static double DGaussian(double x)
        {
            return -2 * x * Math.Pow(Math.E, Math.Pow(-x, 2));
        }
        public static double Bipolar(double x)
        {
            return x < 0 ? -1 : 1;
        }

        public static double BipolarSigmoid(double x)
        {
            return (1 - Math.Exp(-x)) / (1 + Math.Exp(-x));
        }
        public static double DBipolarSigmoid(double x)
        {
            return 0.5 * (1 + BipolarSigmoid(x)) * (1 - BipolarSigmoid(x));
        }

        public static double Scaler(double x, double min, double max)
        {
            return (x - min) / (max - min);
        }
    }

}

