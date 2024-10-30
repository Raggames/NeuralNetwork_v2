using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core
{
    public struct VectorNInputData : IMLInputData
    {
        public double[] Data { get; set; }

        public VectorNInputData(double x, double y)
        {
            Data = new double[] { x, y };
        }

        public VectorNInputData(double x, double y, double z)
        {
            Data = new double[] { x, y, z };
        }
    }
}
