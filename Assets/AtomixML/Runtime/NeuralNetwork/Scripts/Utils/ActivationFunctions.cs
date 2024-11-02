using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assets
{
    public class ActivationFunctions
    {
        public ActivationType activationType;
        public ActivationFunctions(ActivationType type)
        {
            activationType = type;
        }
        public double AFunction(double x)
        {
            switch (activationType)
            {
                case ActivationType.Identity:
                    return Identity(x);
                case ActivationType.BinaryStep:
                    return BinaryStep(x);
                case ActivationType.Logistic:
                    return Logistic(x);
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
        public double ActivationDerivative(double x)
        {
            switch (activationType)
            {
                case ActivationType.Logistic:
                    return DLogistic(x);
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
        public double AFunction(double x, double a)
        {
            switch (activationType)
            {
                case ActivationType.PReLU:
                    return PReLU(x, a);
                case ActivationType.ELU:
                    return ELU(x, a);
            }
            return 0;
        }
        public double ActivationDerivative(double x, double a)
        {
            switch (activationType)
            {
                case ActivationType.PReLU:
                    return DPReLU(x, a);
                case ActivationType.ELU:
                    return DELU(x, a);
            }
            return 0;
        }
        public double Identity(double x)
        {
            return x;
        }

        public double BinaryStep(double x)
        {
            return x < 0 ? 0 : 1;
        }

        public static double Logistic(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));
        }
        public static double DLogistic(double x)
        {
            return Logistic(x) * (1 - Logistic(x));
        }
        public double Tanh(double x)
        {
            return 2 / (1 + Math.Pow(Math.E, -(2 * x))) - 1;
        }
        public double DTanh(double x)
        {
            return 1 - Math.Pow(Tanh(x), 2);
        }
        public double ArcTan(double x)
        {
            return Math.Atan(x);
        }
        public double DArcTan(double x)
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
        public double PReLU(double x, double a)
        {
            return x < 0 ? a * x : x;
        }
        public double DPReLU(double x, double a)
        {
            return x < 0 ? a : 1;
        }
        //Exponential Linear Unit 
        public double ELU(double x, double a)
        {
            return x < 0 ? a * (Math.Pow(Math.E, x) - 1) : x;
        }
        public double DELU(double x, double a)
        {
            return x < 0 ? ELU(x, a) + a : 1;
        }
        public double SoftPlus(double x)
        {
            return Math.Log(Math.Exp(x) + 1);
        }
        public double DSoftPlus(double x)
        {
            return Logistic(x);
        }
        public double BentIdentity(double x)
        {
            return (((Math.Sqrt(Math.Pow(x, 2) + 1)) - 1) / 2) + x;
        }
        public double DBentIdentity(double x)
        {
            return (x / (2 * Math.Sqrt(Math.Pow(x, 2) + 1))) + 1;
        }
        //  public float SoftExponential(float x)
        //  {
        //
        //  }
        public double Sinusoid(double x)
        {
            return Math.Sin(x);
        }
        public double DSinusoid(double x)
        {
            return Math.Cos(x);
        }
        public double Sinc(double x)
        {
            return x == 0 ? 1 : Math.Sin(x) / x;
        }
        public double DSinc(double x)
        {
            return x == 0 ? 0 : (Math.Cos(x) / x) - (Math.Sin(x) / Math.Pow(x, 2));
        }
        public double Gaussian(double x)
        {
            return Math.Pow(Math.E, Math.Pow(-x, 2));
        }
        public double DGaussian(double x)
        {
            return -2 * x * Math.Pow(Math.E, Math.Pow(-x, 2));
        }
        public double Bipolar(double x)
        {
            return x < 0 ? -1 : 1;
        }
        public double BipolarSigmoid(double x)
        {
            return (1 - Math.Exp(-x)) / (1 + Math.Exp(-x));
        }
        public double DBipolarSigmoid(double x)
        {
            return 0.5 * (1 + BipolarSigmoid(x)) * (1 - BipolarSigmoid(x));
        }

        public double Scaler(double x, double min, double max)
        {
            return (x - min) / (max - min);
        }
    }
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
}

