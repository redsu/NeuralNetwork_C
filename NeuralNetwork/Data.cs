using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork {
	class Data {
		public double[][] input;
		public double[][] output;
		Random rand = new Random();
		int numOfVectors = 0;
		int Din = 0;
		int Dout = 0;
		public Data(int input_dimension, int output_dimension, int vector_nums){
			input = new double[vector_nums][];
			output = new double[vector_nums][];
			numOfVectors = vector_nums;
			Din = input_dimension;
			Dout = output_dimension;
			for(int i=0; i<vector_nums; i++)
				input[i] = new double[input_dimension+1];

			for(int i=0; i<vector_nums; i++)
				output[i] = new double[output_dimension];
		}

		public void Shuffle(){
			double temp;
			for(int i=numOfVectors-1; i>=0; i--){
				int pos = rand.Next(i+1);
				for(int j=0; j<Din+1; j++){
					temp = input[i][j];
					input[i][j] = input[pos][j];;
					input[pos][j] = temp;
				}

				for(int j=0; j<Dout; j++){
					temp = output[i][j];
					output[i][j] = output[pos][j];;
					output[pos][j] = temp;
				}
			}
		}
	}
}
