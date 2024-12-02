#include "CMSketch.h"

// #include "CUSketch.h"
// #include "CUSACSketch.h"
// #include "PSketch.h"
// #include "PSACSketch.h"
#include <fstream>
#include <string>

CMSketch *Sketch = new CMSketch(3, 9260);
std::unordered_map<std::string, ull> actual_size;
int main(){
	std::string dat_path = "binary.dat";
	/*Insert your code here using the flowsize interface*/

	std::ifstream file(dat_path,std::ios::binary);
	if(!file){
		std::cout<<"error, file not found!\n";
		return -1;
	}
	unsigned char* buffer = new unsigned char [13];
	memset(buffer,0,13);
	uint packet_count = 0;

	//將每個packet的data讀進來，並insert到sketch
	while(file.read(reinterpret_cast<char*>(buffer),13)|| file.gcount() > 0)
	{
		std::string data(reinterpret_cast<char*>(buffer), file.gcount());
		cuc* constData = buffer;
		Sketch->Insert(constData);
		actual_size[data]++;
		packet_count++;
	}
	file.close();
	file.open(dat_path,std::ios::binary);
	memset(buffer,0,13);
	FILE* out = fopen("result.txt", "w");
	FILE* counters = fopen("counters.txt","w");
	FILE* all_flows = fopen("flows.txt","w");
	float AAE = 0;
	float ARE = 0;

	//for each flows in "actual size" print counter to file, avoid repeated print
 	for (auto it = actual_size.begin(); it != actual_size.end(); ++it) 
	{
        cuc* temp = reinterpret_cast<cuc*>(const_cast<char*>(it->first.c_str()));
		int second = it->second;
		Sketch->PrintCounterFile(temp, second, all_flows);
		AAE += Sketch->CalculateAAE(temp,second);
    }

	//load parameter form pre-trained model
	cuc* path = (unsigned char*)"parameter.txt";
	Sketch->LoadPara(path);
	

	//query by each packet
	//這邊不用print counter，因為會重複print FLOW的counter
	while(file.read(reinterpret_cast<char*>(buffer),13)|| file.gcount() > 0)
	{	
		// packet_num++;
		std::string data(reinterpret_cast<char*>(buffer), 13);
		cuc* constData = buffer;
		ull a = actual_size[data];
		ull query_val;
		query_val = Sketch->Query(constData);
		fprintf(out,"Actual Size: %llu Query Value: %llu\n", a, query_val);
		// AAE+=Sketch->CalculateAAE_ML(constData,a,query_val);
		
		// Sketch->PrintCounterFile(constData,actual_size[data],counters);
	}
	
	AAE /= actual_size.size();

	std::cout<<"AAE: "<<AAE<<std::endl;
	// std::cout<<"AAE per packet: "<<AAE<<std::endl;
	// std::cout<<"ARE: "<<ARE<<std::endl;
	fclose(out);
	fclose(counters);
	return 0;
}