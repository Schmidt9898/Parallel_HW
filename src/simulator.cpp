#include "simulator.hpp"
#include "fmt/core.h"
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>


class Timer {
public:
	std::chrono::time_point<std::chrono::high_resolution_clock> start_p, end;
	std::chrono::duration<float> duration;
	double total_time=0;
	size_t total_byte=0;
	std::string name;
	size_t bytes=0;
	size_t grid=0;

	Timer(std::string name_ = "Timer",size_t bytes_=1) : name(name_),bytes(bytes_){}
	~Timer() {
		//end = std::chrono::high_resolution_clock::now();
		//duration = end - start;
		//std::cout << name << " took: " << duration.count() << "\n";
	}
	void start(){start_p = std::chrono::high_resolution_clock::now();}
	void stop(){
		end = std::chrono::high_resolution_clock::now();
		duration = end - start_p;

		total_byte+=bytes*grid;
		total_time+= double(duration.count());

		//std::cout << name << " took: " << duration.count() << "\n";
	}
	void print()
	{
		std::cout << name << "      " << total_byte*sizeof(Simulator::FloatType)<<"		"<<total_time<<"s	"<< double(total_byte*sizeof(Simulator::FloatType))/1000000000.0/total_time << " GB/s\n";
		//std::cout << name << " bandwith: " << total_byte*sizeof(Simulator::FloatType)<<" "<<total_time << "\n";
	}

};


//Timer sUM("solveUMomentum",4);
//Timer aBU("applyBoundaryU",2);
//Timer sVM("solveVMomentum",3);
//Timer aBV("applyBoundaryV",1);//
//Timer sCEP("solveContinuityEquationP",4);
//Timer aBP("applyBoundaryP",1);//
//Timer cE("calculateError",3);
//Timer iU("iterateU",1);
//Timer iV("iterateV",);
//Timer iP("iterateP",);







void Simulator::setPrinting(bool toPrint) { printing = toPrint; }

void Simulator::initU() {
	for (SizeType i = 0; i <= (grid - 1); i++) {
		u[(i) * (grid + 1) + grid] = 1.0;
		u[(i) * (grid + 1) + grid - 1] = 1.0;
		for (SizeType j = 0; j < (grid - 1); j++) {
			u[(i) * (grid + 1) + j] = 0.0;
		}
	}
}

void Simulator::initV() {
	for (SizeType i = 0; i <= (grid); i++) {
		for (SizeType j = 0; j <= (grid - 1); j++) {
			v[(i)*grid + j] = 0.0;
		}
	}
}

void Simulator::initP() {
	for (SizeType i = 0; i <= (grid); i++) {
		for (SizeType j = 0; j <= (grid); j++) {
			p[(i) * (grid + 1) + j] = 1.0;
		}
	}
}

void Simulator::solveUMomentum(const FloatType Re) { //this updates un
	//un, u, v, p    *  grid-2*grid-1
	//sUM.grid=(grid-2)*(grid-1);
	//sUM.start();
	//exchangeHalo(grid,grid,un.data());
	//#pragma omp parallel for collapse(1)
	for (SizeType i = 1; i <= (grid - 2); i++) {
		//#pragma omp simd
		for (SizeType j = 1; j <= (grid - 1); j++) {
			un[(i) * (grid + 1) + j] = u[(i) * (grid + 1) + j]
				- dt
					* ((u[(i + 1) * (grid + 1) + j] * u[(i + 1) * (grid + 1) + j] - u[(i - 1) * (grid + 1) + j] * u[(i - 1) * (grid + 1) + j]) / 2.0 / dx
					+ 0.25 * ((u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j + 1]) * (v[(i)*grid + j] + v[(i + 1) * grid + j])
							- (u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j - 1]) * (v[(i + 1) * grid + j - 1] + v[(i)*grid + j - 1])) / dy)
					- dt / dx * (p[(i + 1) * (grid + 1) + j] - p[(i) * (grid + 1) + j]) + dt * 1.0 / Re
					* ((u[(i + 1) * (grid + 1) + j] - 2.0 * u[(i) * (grid + 1) + j] + u[(i - 1) * (grid + 1) + j]) / dx / dx
					 + (u[(i) * (grid + 1) + j + 1] - 2.0 * u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j - 1]) / dy / dy);
		}
	}
	//sUM.stop();
}

void Simulator::applyBoundaryU() {//this updates un
	//un * grid-1  *2
	//aBU.grid=grid-1  *2;
	//aBU.start();
	//#pragma omp parallel for collapse(1)




	if(coords[1]==0)
	for (SizeType j = 1; j <= (grid - 1); j++) {
		//un[(0) * (grid + 1) + j] = 0.0;
		un[j] = 0.0;// WELL, is this optimizing ?
		}
	if(coords[1]==dims[1])
	for (SizeType j = 1; j <= (grid - 1); j++) {
		un[(grid - 1) * (grid + 1) + j] = 0.0;
	}
	//un *grid
	//#pragma omp parallel for collapse(1)
	if(coords[0]==0)
	for (SizeType i = 0; i <= (grid - 1); i++) {
		un[(i) * (grid + 1) + 0] = -un[(i) * (grid + 1) + 1];}
	if(coords[0]==dims[0])
	for (SizeType i = 0; i <= (grid - 1); i++) {
		un[(i) * (grid + 1) + grid] = 2 - un[(i) * (grid + 1) + grid - 1];
	}
	//aBU.stop();

}

void Simulator::solveVMomentum(const FloatType Re) {
	//vn  v  u  *grid-1
	//sVM.grid=grid-1;
	//sVM.start();
	//#pragma omp parallel for collapse(1)
	for (SizeType i = 1; i <= (grid - 1); i++) {
		//#pragma omp simd
		for (SizeType j = 1; j <= (grid - 2); j++) {
			vn[(i)*grid + j] = v[(i)*grid + j]
				- dt * (0.25 * ((u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j + 1]) * (v[(i)*grid + j] + v[(i + 1) * grid + j])
							  - (u[(i - 1) * (grid + 1) + j] + u[(i - 1) * (grid + 1) + j + 1]) * (v[(i)*grid + j] + v[(i - 1) * grid + j])) / dx
							  + (v[(i)*grid + j + 1] * v[(i)*grid + j + 1] - v[(i)*grid + j - 1] * v[(i)*grid + j - 1]) / 2.0 / dy)
							  - dt / dy * (p[(i) * (grid + 1) + j + 1] - p[(i) * (grid + 1) + j]) + dt * 1.0 / Re
							  * ((v[(i + 1) * grid + j] - 2.0 * v[(i)*grid + j] + v[(i - 1) * grid + j]) / dx / dx
							  + (v[(i)*grid + j + 1] - 2.0 * v[(i)*grid + j] + v[(i)*grid + j - 1]) / dy / dy);
		}
	}
	//sVM.stop();
}

void Simulator::applyBoundaryV() {
	//aBV.grid=grid-2;
	//aBV.start();
	//#pragma omp parallel for collapse(1)
	
	if(coords[1]==0)
	for (SizeType j = 1; j <= (grid - 2); j++) {
		vn[(0) * grid + j] = -vn[(1) * grid + j];}
	if(coords[1]==dims[1])
	for (SizeType j = 1; j <= (grid - 2); j++) {
		vn[(grid)*grid + j] = -vn[(grid - 1) * grid + j];
	}
	//aBV.stop();
	//aBV.grid=grid+1;
	//aBV.start();
	//#pragma omp parallel for collapse(1)
	if(coords[0]==0)
	for (SizeType i = 0; i <= (grid); i++) {
		vn[(i)*grid + 0] = 0.0;}
	if(coords[0]==dims[0])
	for (SizeType i = 0; i <= (grid); i++) {
		vn[(i)*grid + grid - 1] = 0.0;
	}
	//aBV.stop();
}

void Simulator::solveContinuityEquationP(const FloatType delta) {
	//pn,p,un,vn grid-1*grid-1
	//sCEP.grid=(grid-1)*(grid-1);
	//sCEP.start();
	//#pragma omp parallel for collapse(1)
	//exchangeHalo(grid,grid,un.data());
	//exchangeHalo(grid,grid,vn.data());
	
	for (SizeType i = 1; i <= (grid - 1); i++) {
		for (SizeType j = 1; j <= (grid - 1); j++) {
			pn[(i) * (grid + 1) + j] = p[(i) * (grid + 1) + j]
				- dt * delta * ((un[(i) * (grid + 1) + j] - un[(i - 1) * (grid + 1) + j]) / dx + (vn[(i)*grid + j] - vn[(i)*grid + j - 1]) / dy);
		}
	}
	//sCEP.stop();
}

//.grid=grid;
//.start();
//.stop();




void Simulator::applyBoundaryP() {
	//aBP.grid=grid-1;
	//aBP.start();
	//#pragma omp parallel for collapse(1)
	if(coords[1]==0)
		for (SizeType i = 1; i <= (grid - 1); i++) {
			pn[(i) * (grid + 1) + 0] = pn[(i) * (grid + 1) + 1];
	}
	if(coords[1]==dims[1]-1)
		for (SizeType i = 1; i <= (grid - 1); i++) {
			pn[(i) * (grid + 1) + grid] = pn[(i) * (grid + 1) + grid - 1];
	}
	//aBP.stop();
	//aBP.grid=grid+1;
	//aBP.start();
	//#pragma omp parallel for collapse(1)
	if(coords[0]==0)
		for (SizeType j = 0; j <= (grid); j++) {
			pn[(0) * (grid + 1) + j] = pn[(1) * (grid + 1) + j];
	}
	if(coords[0]==dims[0]-1)
		for (SizeType j = 0; j <= (grid); j++) {
			pn[(grid) * (grid + 1) + j] = pn[(grid - 1) * (grid + 1) + j];
	}
	//aBP.stop();
}

Simulator::FloatType Simulator::calculateError() {
	//m,un,vn grid-1*grid-1
	//cE.grid=(grid-1)*(grid-1);
	//cE.start();
	FloatType error = 0.0;
	//#pragma omp parallel for collapse(1) reduction(+:error)
	for (SizeType i = 1; i <= (grid - 1); i++) {
		FloatType p_error = 0.0;
		//#pragma omp simd
		for (SizeType j = 1; j <= (grid - 1); j++) {
			m[(i) * (grid + 1) + j] =
				((un[(i) * (grid + 1) + j] - un[(i - 1) * (grid + 1) + j]) / dx + (vn[(i)*grid + j] - vn[(i)*grid + j - 1]) / dy);
		p_error += fabs(m[(i) * (grid + 1) + j]);
			}
		error += p_error;
	}

	//fmt::print("Error is {} <--\n", error);

	double true_error = 0.0;
	MPI_Allreduce(&error , &true_error , 1 , MPI_DOUBLE , MPI_SUM  , MPI_COMM_WORLD);
	//MPI_Reduce(&error , &true_error , 1 , MPI_DOUBLE , MPI_SUM , 0 , MPI_COMM_WORLD);
	//cE.stop();

	return true_error;
}

void Simulator::iterateU() {
	std::swap(u, un);
	// for (SizeType i = 0; i <= (grid - 1); i++) {
	//     for (SizeType j = 0; j <= (grid); j++) {
	//         u[(i) * (grid + 1) + j] = un[(i) * (grid + 1) + j];
	//     }
	// }
}

void Simulator::iterateV() {
	std::swap(v, vn);
	// for (SizeType i = 0; i <= (grid); i++) {
	//     for (SizeType j = 0; j <= (grid - 1); j++) {
	//         v[(i)*grid + j] = vn[(i)*grid + j];
	//     }
	// }
}

void Simulator::iterateP() {
	std::swap(p, pn);
	// for (SizeType i = 0; i <= (grid); i++) {
	//     for (SizeType j = 0; j <= (grid); j++) {
	//         p[(i) * (grid + 1) + j] = pn[(i) * (grid + 1) + j];
	//     }
	// }
}

void Simulator::deallocate() {
	// it doesn't do anything until we use vectors
	// because that deallocates automatically
	// but if we have to use a more raw data structure later it is needed
	// and when the the Tests overwrites some member those might won't deallocate
	MPI_Finalize();
}

Simulator::Simulator(SizeType gridP)
	: grid([](auto g) {
		  if (g <= 1) {
			  throw std::runtime_error("Grid is smaller or equal to 1.0, give larger number");
		  }
		  int gx=int(g);
		  int gy=int(g);
		  MPISetup(&gx, &gy);
		  return SizeType(gx);
	  }(gridP)),
	  dx(1.0 / static_cast<FloatType>(grid - 1)),
	  dy(1.0 / static_cast<FloatType>(grid - 1)),
	  dt(0.001 / std::pow(grid / 128.0 * 2.0, 2.0)),
	  u	((grid + 1) * (grid + 1)),
	  un((grid + 1) * (grid + 1)),
	  v	((grid + 1) * (grid + 1)),
	  vn((grid + 1) * (grid + 1)),
	  p	((grid + 1) * (grid + 1)),
	  pn((grid + 1) * (grid + 1)),
	  m	((grid + 1) * (grid + 1)) {
    //fmt::print("The size of the grid is {}\n", grid);
	initU();
	initV();
	initP();

}

void Simulator::run(const FloatType delta, const FloatType Re, unsigned maxSteps) {
	if (printing) {
		fmt::print("Running simulation with delta: {}, Re: {}\n", delta, Re);
	}
	auto error = std::numeric_limits<FloatType>::max();
	unsigned step = 1;
	while (error > 0.00000001 && step <= maxSteps) {
		solveUMomentum(Re);
		applyBoundaryU();

		solveVMomentum(Re);
		applyBoundaryV();
		exchangeHalo(grid,grid,un.data());
		exchangeHalo(grid,grid,vn.data());

		solveContinuityEquationP(delta);
		applyBoundaryP();

		error = calculateError();

		if (my_rank==0 && printing && (step % 1000 == 1)) {
			fmt::print("Error is {} for the step {}\n", error, step);
		}

		iterateU();
		iterateV();
		iterateP();
		exchangeHalo(grid,grid,u.data());
		exchangeHalo(grid,grid,v.data());
		exchangeHalo(grid,grid,p.data());

		++step;
	}
	//std::cout<<"Name 		    Count 		Time 		GB/s\n";
	//sUM .print();
	//aBU .print();
	//sVM .print();
	//aBV .print();
	//sCEP.print();
	//aBP .print();
	//cE  .print();


}

Simulator::~Simulator() { deallocate(); }