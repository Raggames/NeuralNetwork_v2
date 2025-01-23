using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;


namespace Atom.MachineLearning.Core.Training
{
    /// <summary>
    /// Abstraction d'un trainer
    /// Un mod�le/algorithme va toujours de paire avec son ITrainer, qui impl�mente toutes les fonctions d'apprentissage
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IMLTrainer<TModel, TModelInputData, TModelOutputData>
        where TModel : IMLModel<TModelInputData, TModelOutputData>
    {
        public TModel trainedModel { get; set; }

        /// <summary>
        /// Train the model 
        /// </summary>
        /// <param name="model"></param>
        /// <param name="x_datas"></param>
        /// <returns></returns>
        public Task<ITrainingResult> Fit(TModelInputData[] x_datas);

        public ITrainingResult FitSynchronously(TModelInputData[] x_datas);

        /// <summary>
        /// Return the accuracy of the model after running a test train
        /// </summary>
        /// <param name="model"></param>
        /// <param name="x_datas"></param>
        /// <returns></returns>
        public Task<double> Score();

        public double ScoreSynchronously();
    }

    /// <summary>
    /// Abstraction d'un trainer
    /// Un mod�le/algorithme va toujours de paire avec son ITrainer, qui impl�mente toutes les fonctions d'apprentissage
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IMLSupervisedTrainer<TModel, TModelInputData, TModelOutputData>
        where TModel : IMLModel<TModelInputData, TModelOutputData>
    {
        public TModel trainedModel { get; set; }

        public Task<ITrainingResult> Fit(TModelInputData[] x_datas, TModelOutputData[] t_datas);

        public ITrainingResult FitSynchronously(TModelInputData[] x_datas, TModelOutputData[] t_datas);

        /// <summary>
        /// Return the accuracy of the model after running a test train
        /// </summary>
        /// <param name="model"></param>
        /// <param name="x_datas"></param>
        /// <returns></returns>
        public Task<double> Score();

        public double ScoreSynchronously();

    }

    /*public interface IMLSerializationContext
    {
        /// <summary>
        /// Saves the model after fitting
        /// </summary>
        /// <param name="outputFilename"></param>
        public void Save(string outputFilename);

        /// <summary>
        /// Load the model from a filename 
        /// </summary>
        /// <param name="filename"></param>
        public void Load(string filename);
    }*/
}
