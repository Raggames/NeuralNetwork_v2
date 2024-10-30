using Assets.AtomixML.Core.Training;
using Atom.MachineLearning.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Unsupervised.KMeanClustering
{
    public class KMeanClusteringTrainer : IMLTrainer<KMeanClusteringModel, VectorNInputData, ClassificationOutputData, UnsupervisedClassificationVectorNDataSet<VectorNInputData>>
    {
        public async Task<ITrainingResult> Fit(KMeanClusteringModel model, UnsupervisedClassificationVectorNDataSet<VectorNInputData> trainingDatas)
        {
            throw new NotImplementedException();
        }
    }
}
