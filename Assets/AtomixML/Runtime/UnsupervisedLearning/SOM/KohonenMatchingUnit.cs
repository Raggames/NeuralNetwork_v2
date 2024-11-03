using Atom.MachineLearning.Core;

namespace Atom.MachineLearning.Unsupervised.SelfOrganizingMap
{
    public struct KohonenMatchingUnit : IMLInOutData
    {
        public int XCoordinate { get; set; }
        public int YCoordinate { get; set; }

        public NVector WeightVector { get; set; }

        /// <summary>
        /// 0 in the case of a prediction (best matchin unit)
        /// Computed distance
        /// </summary>
        public double Distance { get; set; }
    }
}
