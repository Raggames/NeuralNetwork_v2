using System.Threading.Tasks;
using UnityEngine;


namespace Atom.MachineLearning.Core
{
    /// <summary>
    /// Abstraction d'un trainer
    /// Un modèle/algorithme va toujours de paire avec son ITrainer, qui implémente toutes les fonctions d'apprentissage
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IMLTrainer<TModel, TModelInputData, TModelOutputData, ITrainingDataSet>
        where TModelInputData : IMLInputData
        where TModelOutputData : IMLOutputData
        where TModel : IMLModel<TModelInputData, TModelOutputData>
        where ITrainingDataSet : IMLTrainingDataSet<TModelInputData>
    {
        public int Epochs { get; set; }

        public int currentEpoch { get; }

        public Task<ITrainingResult> Fit(TModel model, ITrainingDataSet trainingDatas);
    }
}
