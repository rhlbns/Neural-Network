package DeepLearn;

import java.util.Random;

import Jama.Matrix;

public class NeuralNetwork {
	int Nlayers;
	int[] Nnodes;
	float alpha;
	Matrix[] Weights;
	Matrix[] DelJ;
	Matrix[] DelX,Y;
	
	public NeuralNetwork(int[] Nnodes, float alpha) {
		this.Nlayers= Nnodes.length;
		this.Nnodes= Nnodes;
		this.alpha= alpha;
		Weights= new Matrix[Nnodes.length-1];
		DelJ= new Matrix[Nnodes.length-1];
		DelX= new Matrix[Nnodes.length];
		Y= new Matrix[Nnodes.length];
		
		Random random= new Random();
		int row,column;
		Matrix M= null;
		row= Nnodes[1];
		column= Nnodes[0];
		M= new Matrix(row,column);
		for(int ii=0;ii<row;ii++) {
			for(int jj=0;jj<column;jj++) {
				M.set(ii, jj, random.nextFloat());
			}
		}
		Weights[0]= M;
		
		for(int j=0;j<Nnodes.length-2;j++) {
			row= Nnodes[j+2];
			column= Nnodes[j+1];
			M= new Matrix(row,column);
			for(int ii=0;ii<row;ii++) {
				for(int jj=0;jj<column;jj++) {
					M.set(ii, jj, random.nextFloat());
				}
			}
			Weights[j+1]= M;
			
		}
	}
	
	public Matrix FeedForward(Matrix input) {
		Matrix M= input;
		Y[0]= input;
		for(int i=0;i<Nlayers-1;i++) {
			Y[i+1]= Weights[i].times(M);
			Y[i+1]= sigmoid(Y[i+1]);
			M= Y[i+1];
		}
		return M;
	}
	
	public void BackPropagation(Matrix del) {
		Matrix delx,delj;
		int row= Weights[Nlayers-2].getRowDimension();
		int column= Weights[Nlayers-2].getColumnDimension();
		double value,sum;
		delj= new Matrix(row,column);
		delx= new Matrix(column,1);
		for(int i=0;i<row;i++) {
			sum= del.get(i, 0)*Y[Nlayers-1].get(i, 0)*(1-Y[Nlayers-1].get(i, 0));
			for(int j=0;j<column;j++) {
				value= sum*Y[Nlayers-2].get(j, 0);
				delj.set(i, j, value);
			}
			delx.set(i, 0, sum);
		}
		DelX[Nlayers-1]= delx;
		DelJ[Nlayers-2]= delj;
		
		for(int i=Nlayers-3;i>=0;i--) {
			row= Weights[i].getRowDimension();
			column= Weights[i].getColumnDimension();
			delx= new Matrix(row,1);
			delj= new Matrix(row,column);
			for(int ii=0;ii<row;ii++) {
				sum= 0;
				for(int jj=0;jj<Y[i+2].getRowDimension();jj++) {
					sum+= DelX[i+2].get(jj, 0)*Weights[i+1].get(jj, ii);
				}
				sum= sum*Y[i+1].get(ii, 0)*(1-Y[i+1].get(ii, 0));
				delx.set(ii, 0, sum);
				
				for(int jj=0;jj<column;jj++) {
					value= sum*Y[i].get(jj, 0);
					delj.set(ii, jj, value);
				}
			}
			DelJ[i]= delj;
			DelX[i+1]= delx;
		}
		
		for(int i=0;i<Nlayers-1;i++) {
			row= Weights[i].getRowDimension();
			column= Weights[i].getColumnDimension();
			for(int ii=0;ii<row;ii++) {
				for(int jj=0;jj<column;jj++) {
					value= Weights[i].get(ii, jj)-alpha*DelJ[i].get(ii, jj);
					Weights[i].set(ii, jj, value);
				}
			}
		}
	}
	
	public Matrix sigmoid(Matrix X) {
		int row= X.getRowDimension();
		int column= X.getColumnDimension();
		Matrix M= new Matrix(row,column);
		double y;
		for(int i=0;i<row;i++) {
			for(int j=0;j<column;j++) {
				y= 1/(1+Math.exp(-X.get(i, j)));
				M.set(i, j, y);
			}
		}
		return M;
	}
	
}
