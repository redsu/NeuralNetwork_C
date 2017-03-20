using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.ComponentModel;


namespace NeuralNetwork {
	class NeuralNetwork {
		Random rand = new Random();
		public List<Layer> layer = new List<Layer>();
		int numOfLayer = 0;
		Data NN_Data;
		bool[] validationDataList;
		int input_dimension;
		int output_dimension;
		int vector_nums;
		int input_width;
		int iterationLimit = 100;
		double eta = 0.7;
		double learningRate = 0.9990;
		double EoutRMS = 0.0, EinRMS = 0.0;
		double Eout01 = 0.0;
		double sampleRate = Math.Round(1.0/3.0, 4);
		double correctness = 0.0;
		double einCorrectness = 0.0;
		EtaType eta_type = EtaType.LearningRate;
		double alpha = 0.0;
		double current_best_correctness = 0.0;
		
		[Browsable(false)]
		public double ErrOut01{
			get{return Eout01;}
		}

		[Browsable(false)]
		public double ErrOutRMS{
			get{return EoutRMS;}
		}

		[Browsable(false)]
		public double ErrInRMS{
			get{return EinRMS;}
		}
		
		[Browsable(false)]
		public int NumOfLayer{
			get{return numOfLayer;}
			set{numOfLayer = value;}
		}

		[Browsable(false)]
		public int Dimout{
			get{return output_dimension;}
			set{output_dimension = value;}
		}

		[Browsable(false)]
		public double Correctness{
			get{return correctness;}
		}

		[Browsable(false)]
		public double EinCorrectness{
			get{return einCorrectness;}
		}

		[Browsable(false)]
		public double CurrentBestOutCorrectness{
			get{return current_best_correctness;}
		}

		[Category("NN Setting"), Description("")]
		public double Eta{
			get{return eta;}
			set{eta = value;}
		}

		[Category("NN Setting"), Description("")]
		public double SampleRate{
			get{return sampleRate;}
			set{sampleRate = Math.Round(value,4);}
		}

		[Category("NN Setting"), Description("")]
		public int IterationLimit{
			get{return iterationLimit;}
			set{iterationLimit = value;}
		}

		[Category("NN Setting"), Description("")]
		public double LearningRate{
			get{return learningRate;}
			set{learningRate = Math.Round(value, 4);}
		}
		[Category("NN Setting"), Description("")]
		public EtaType Eta_type{
			get{return eta_type;}
			set{eta_type = value;}
		}

		[Category("NN Setting"), Description("")]
		public double Alpha{
			get{return alpha;}
			set{alpha = value;}
		}

		public NeuralNetwork(){
			;
		}

		public bool ImportData(){
			OpenFileDialog openfile = new OpenFileDialog();
			System.IO.StreamReader File = null;
			string fileString = "";
			string[] elements;
			char[] separator = new char[] {' ',',',';'};
			double[] min, max;

			//Clear layer list
			layer.Clear();

			//If file exists.
			if (openfile.ShowDialog() == DialogResult.OK && openfile.FileName != ""){
				File = new System.IO.StreamReader(openfile.FileName);

				//If file isn't broken
				if(File != null){
					fileString = File.ReadLine().Trim();
					elements = fileString.Split(separator);
					vector_nums		 = int.Parse(elements[0]);
					input_dimension	 = int.Parse(elements[1]);
					output_dimension = int.Parse(elements[2]);
					input_width		 = int.Parse(elements[3]);

					min = new double[input_dimension+output_dimension];
					max = new double[input_dimension+output_dimension];

					for(int i=0; i<input_dimension+output_dimension; i++){
						min[i] = double.MaxValue;
						max[i] = double.MinValue;
					}

					//I:inputdim, O:outputDim, H:hiddenDim
					numOfLayer = 3;	//Default layers : 3(1 hidden)
					//input dimension / hidden layer has 5 Neurons / output dimension
					layer.Add(new Layer(input_dimension+1, 10));
					layer.Add(new Layer(11, output_dimension));
					layer.Add(new Layer(output_dimension+1, 1));
					for(int i=0; i<numOfLayer; i++)
						layer[i].Node = i;

					//new a Data class to save the input data
					NN_Data = new Data(input_dimension, output_dimension, vector_nums);
					
					//initiate validationDataList; True:testing / False:training
					validationDataList = new bool[vector_nums];

					for(int i=0; i<vector_nums; i++){
						//initiate validationDataList; True:testing / False:training
						validationDataList[i] = false;

						//Read in the data line by line
						fileString = File.ReadLine().Trim();
						elements = fileString.Split(separator);
						for(int j=1; j<input_dimension+1; j++){
							NN_Data.input[i][j] = double.Parse(elements[j-1]);
							if(NN_Data.input[i][j]<min[j-1])
								min[j-1] = NN_Data.input[i][j];
							if(NN_Data.input[i][j]>max[j-1])
								max[j-1] = NN_Data.input[i][j];
						}
						NN_Data.input[i][0] = 1.0;

						//Find the maximum and minimum of every dimension for normalization
						for(int j=input_dimension; j<input_dimension+output_dimension; j++){
							NN_Data.output[i][j-input_dimension] = double.Parse(elements[j]);
							if(NN_Data.output[i][j-input_dimension]<min[j])
								min[j] = NN_Data.output[i][j-input_dimension];
							if(NN_Data.output[i][j-input_dimension]>max[j])
								max[j] = NN_Data.output[i][j-input_dimension];
						}
					}

					for(int i=0; i<vector_nums; i++){
						//make sure that the value of (maximum-minimum) > 0
						//normalize the data dimension by dimension
						for(int j=1; j<input_dimension+1; j++)
							if((max[j-1]-min[j-1])>0.000001)
								NN_Data.input[i][j] = (NN_Data.input[i][j]) / (max[j-1]-min[j-1]);
						for(int j=input_dimension; j<input_dimension+output_dimension; j++)
							if((max[j]-min[j])>0.000001)
								NN_Data.output[i][j-input_dimension] = (NN_Data.output[i][j-input_dimension]) / (max[j]-min[j]);
					}
					//Shuffle the data for random sampling
					NN_Data.Shuffle();
					//label the testing data
					BuildValidList();
					return true;
				}
				return false;
			}
			else
				return false;
		}

		public void InsertHiddenLayer(int position, int numOfNeurons){
			if(position > 0 && position < numOfLayer){
				Layer newLayer = null;
				//Insert a new layer and make sure the input / output dimension are matched
				newLayer = new Layer(numOfNeurons+1, layer[position].InputDim-1);
				layer[position-1].OutputDim = numOfNeurons;
				layer[position-1].Reset();
				layer.Insert(position, newLayer);
				numOfLayer++;
				for(int i=0; i<numOfLayer; i++)
					layer[i].Node = i;
				Reset();
			}
		}

		public void DeleteHiddenLayer(int position){
			//Hidden layer only can be deleted when we have more than 1 hidden layer
			if(position > 0 && position < numOfLayer-1 && numOfLayer > 3){
				//Delete the selected layer and make sure the input / output dimension are matched
				layer[position-1].OutputDim = layer[position+1].InputDim-1;
				layer[position-1].Reset();
				layer.RemoveAt(position);
				numOfLayer--;
				for(int i=0; i<numOfLayer; i++)
					layer[i].Node = i;
				Reset();
			}
		}

		public void EditHiddenLayer(int position, int numofperc){
			//Add or delete a neuron in chosen layer
			if(position > 0 && position <= numOfLayer-1){
				layer[position].InputDim = numofperc+1;
				layer[position].Reset();
				layer[position-1].OutputDim = numofperc;
				layer[position-1].Reset();
				for(int i=0; i<numOfLayer; i++)
					layer[i].Node = i;
				Reset();
			}
		}

		public void UpdateNN(){
			double Ein_iter = 0.0;
			int numOfTrainData = 0;
			int N = (int)(vector_nums*sampleRate);
			//Update NN system
			for(int i=0; i<vector_nums-N; i++){
				//if the data is not label as testing data
				if(!validationDataList[i]){
					numOfTrainData++;
					//Update all x (Pass forward)
					for(int j=0; j<layer[0].InputDim; j++)
						layer[0].Neurons[j] = NN_Data.input[i][j];

                    for (int j = 1; j < numOfLayer; j++)
                    {
                        if (j != numOfLayer-1)
                            layer[j].Neurons = (double[])layer[j - 1].Output(false).Clone();
                        else
                            layer[j].Neurons = (double[])layer[j - 1].Output(true).Clone();
                    }
				
					//Update all epsilons
					layer[numOfLayer-1].UpdateLastLayerEpsilon(NN_Data.output[i]);
					for(int j=numOfLayer-2; j>=0; j--)
						layer[j].UpdateEpsilon(layer[j+1].Epsilons);
				
					////Update all weights
					//for(int j=0; j<numOfLayer-1; j++)
                    for (int j = numOfLayer - 2; j >= 0; j--)
						layer[j].UpdateWeight(layer[j+1].Epsilons, eta, alpha);

					//Calculate RMS Error
					for(int j=1; j<layer[numOfLayer-1].InputDim; j++)
						Ein_iter += Math.Pow((NN_Data.output[i][j-1]-layer[numOfLayer-1].Neurons[j]), 2.0);
				}
			}
			//Calculate average RMS Error
            EinRMS = Math.Sqrt(Ein_iter / ((double)numOfTrainData * ((double)layer[numOfLayer - 1].InputDim - 1.0)));
			EinValidation();
			ValidationTable();
			switch(eta_type){
				//Define eta type
				case EtaType.LearningRate://Learning Rate Type
					eta *= learningRate;
					break;
				case EtaType.Logistic://Logistic Function with EinRMS Type
					eta = 2.0 / (1.0 + Math.Exp(-EinRMS))-1.0;
					break;
				case EtaType.InverseLogistic://Logistic Function with EinCorrectness Type
					eta = 2.0 / (1.0 + Math.Exp(-einCorrectness))-1.0;
					break;
				case EtaType.HypoTangent://Logistic Function with EinCorrectness Type
					eta = Math.Tanh(EinRMS);
					break;
				
			}			
			if(correctness>current_best_correctness)
				current_best_correctness = correctness;
		}

		public void BuildValidList(){
			int N = (int)(vector_nums*sampleRate);

			//label the last N data as testing data
			//It is an random set since that we shuffle the data first
			for(int i=vector_nums-1; i>=vector_nums-N; i--)
				validationDataList[i] = true;			
		}
		
		public void Validation(){
			double Eout_test = 0.0;
			int numOfTestData = (int)(vector_nums*sampleRate);

			for(int i=vector_nums-1; i>=vector_nums-numOfTestData; i--){
				if(validationDataList[i]){
					//Update all x (Pass forward)
					for(int j=0; j<layer[0].InputDim; j++)
						layer[0].Neurons[j] = NN_Data.input[i][j];

                    for (int j = 1; j < numOfLayer; j++)
                    {
                        //layer[j].Neurons = (double[])layer[j - 1].Output().Clone();
                        if (j != numOfLayer - 1)
                            layer[j].Neurons = (double[])layer[j - 1].Output(false).Clone();
                        else
                            layer[j].Neurons = (double[])layer[j - 1].Output(true).Clone();
                    }

					//Calculate RMS Error of testing data
					for(int j=1; j<layer[numOfLayer-1].InputDim; j++)
						Eout_test += Math.Pow((NN_Data.output[i][j-1]-layer[numOfLayer-1].Neurons[j]), 2.0);
				}
			}
			EoutRMS = Math.Sqrt(Eout_test/((double)numOfTestData*(layer[numOfLayer-1].InputDim-1.0)));
		}

		public int[][] ValidationTable(){
			int[][] table = new int[output_dimension][];
			for(int i=0; i<output_dimension; i++){
				table[i] = new int[output_dimension];
				for(int j=0; j<output_dimension; j++)
					table[i][j] = 0;
			}

			int numOfTestData = (int)(vector_nums*sampleRate);
			double Max;
			int indexV = 0, indexY = 0;
			correctness = 0.0;

			for(int i=vector_nums-1; i>=vector_nums-numOfTestData; i--){
				if(validationDataList[i]){
					//Update all x (Pass forward)
					for(int j=0; j<layer[0].InputDim; j++)
						layer[0].Neurons[j] = NN_Data.input[i][j];

                    for (int j = 1; j < numOfLayer; j++)
                    {
                        //layer[j].Neurons = (double[])layer[j - 1].Output().Clone();
                        if (j != numOfLayer - 1)
                            layer[j].Neurons = (double[])layer[j - 1].Output(false).Clone();
                        else
                            layer[j].Neurons = (double[])layer[j - 1].Output(true).Clone();
                    }

					//Find the dimesion has largest value(the class which the input be classified)
					Max = double.MinValue;
					for(int j=1; j<layer[numOfLayer-1].InputDim; j++){
						if(Max < layer[numOfLayer-1].Neurons[j]){
							Max = layer[numOfLayer-1].Neurons[j];
							indexY = j-1;
						}
					}

					Max = double.MinValue;
					for(int j=0; j<layer[numOfLayer-1].InputDim-1; j++){
						if(Max < NN_Data.output[i][j]){
							Max = NN_Data.output[i][j];
							indexV = j;
						}
					}
					table[indexY][indexV]++;
					//if the input is classified correctly, correctness+1
					if(indexV == indexY)
						correctness += 1.0;
				}
			}
			//Calculate the 0/1 correctness of testing data
			correctness /= (double)((int)(vector_nums*sampleRate));
			return table;
		}

		public void EinValidation(){
			int numOfTestData = (int)(vector_nums*sampleRate);
			double Max;
			int indexV = 0, indexY = 0;
			einCorrectness = 0.0;
			for(int i=0; i<vector_nums-numOfTestData; i++){
				
				if(!validationDataList[i]){
					//Update all x (Pass forward)
					for(int j=0; j<layer[0].InputDim; j++)
						layer[0].Neurons[j] = NN_Data.input[i][j];

                    for (int j = 1; j < numOfLayer; j++)
                    {
                        //layer[j].Neurons = (double[])layer[j - 1].Output().Clone();
                        if (j != numOfLayer - 1)
                            layer[j].Neurons = (double[])layer[j - 1].Output(false).Clone();
                        else
                            layer[j].Neurons = (double[])layer[j - 1].Output(true).Clone();
                    }

					//Find the dimesion has largest value(the class which the input be classified)
					Max = double.MinValue;
					for(int j=1; j<layer[numOfLayer-1].InputDim; j++){
						if(Max < layer[numOfLayer-1].Neurons[j]){
							Max = layer[numOfLayer-1].Neurons[j];
							indexY = j-1;
						}
					}

					Max = double.MinValue;
					for(int j=0; j<layer[numOfLayer-1].InputDim-1; j++){
						if(Max < NN_Data.output[i][j]){
							Max = NN_Data.output[i][j];
							indexV = j;
						}
					}

					//if the input is classified correctly, correctness+1
					if(indexV == indexY)
						einCorrectness += 1.0;
				}
			}
			//Calculate the 0/1 correctness of training data
			einCorrectness /= (double)(vector_nums - (int)(vector_nums*sampleRate));
		}

		public void Reset(){
			switch(eta_type){
				//Define eta type
				case EtaType.LearningRate://Learning Rate Type
					eta *= learningRate;
					break;
				case EtaType.Logistic://Logistic Function with EinRMS Type
					eta = 2.0 / (1.0 + Math.Exp(-EinRMS))-1.0;
					break;
				case EtaType.InverseLogistic://Logistic Function with EinCorrectness Type
					eta = 2.0 / (1.0 + Math.Exp(-einCorrectness))-1.0;
					break;
				case EtaType.HypoTangent://Logistic Function with EinCorrectness Type
					eta = Math.Tanh(EinRMS);
					break;
			}
			//initialize current_best_correctness
			current_best_correctness = 0.0;

			//initialize all the weights
			for(int i=0; i<numOfLayer; i++)
				layer[i].InitWeights();
			BuildValidList();
		}
	}
	public enum EtaType{LearningRate, Logistic, InverseLogistic, HypoTangent};
}
