using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ComplexNum{
	class Complex{
		public double a, b;
		public Complex(){
			a = b = 0.0;
		}
		public Complex(double Re, double Im){
			a = Re;
			b = Im;
		}
		public static Complex operator +(Complex c1, Complex c2){  
			return new Complex(c1.a + c2.a, c1.b + c2.b);  
		}
		public static Complex operator -(Complex c1, Complex c2){  
			return new Complex(c1.a - c2.a, c1.b - c2.b);  
		} 
		public static Complex operator *(Complex c1, Complex c2){  
			return new Complex(c1.a * c2.a - c1.b * c2.b, c1.a * c2.b + c1.b * c2.a);
		}
		public static Complex operator /(Complex c1, Complex c2){  
			double scalar = c2.a*c2.a+c2.b*c2.b;
			return new Complex((c1.a * c2.a + c1.b * c2.b)/scalar, (-c1.a * c2.b + c1.b * c2.a)/scalar);
		} 
		public override string ToString() {
			if(b>=0)
				return a.ToString()+"+"+b.ToString()+"i";
			else
				return a.ToString()+b.ToString()+"i";
		}
	}
}
