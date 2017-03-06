using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ComplexNum;

namespace NeuralNetwork_C{
	class Data_C {
		public Complex[][] input;
        public Complex[][] output;
		Random rand = new Random();
		int numOfVectors = 0;
		int Din = 0;
		int Dout = 0;
		public Data_C(int input_dimension, int output_dimension, int vector_nums){
            input = new Complex[vector_nums][];
            output = new Complex[vector_nums][];
			numOfVectors = vector_nums;
			Din = input_dimension;
			Dout = output_dimension;
            for (int i = 0; i < vector_nums; i++)
            {
                input[i] = new Complex[input_dimension + 1];
                for (int j = 0; j < input_dimension+1; j++)
                {
                    input[i][j] = new Complex();
                }
            }

            input[1][0].a = 10.0;
            for (int i = 0; i < vector_nums; i++)
            {
                output[i] = new Complex[output_dimension];
                for (int j = 0; j < output_dimension; j++)
                {
                    output[i][j] = new Complex();
                }
            }
		}

		public void Shuffle(){
            Complex temp;
			for(int i=numOfVectors-1; i>=0; i--){
				int pos = rand.Next(i+1);
				for(int j=0; j<Din+1; j++){
					temp = input[i][j];
					input[i][j] = input[pos][j];
					input[pos][j] = temp;
				}

				for(int j=0; j<Dout; j++){
					temp = output[i][j];
					output[i][j] = output[pos][j];
					output[pos][j] = temp;
				}
			}
		}
	}
}
