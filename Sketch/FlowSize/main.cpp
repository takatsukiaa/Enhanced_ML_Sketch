#include "CMSketch.h"

// #include "CUSketch.h"
// #include "CUSACSketch.h"
// #include "PSketch.h"
// #include "PSACSketch.h"
#include <fstream>
#include <string>
#define MICE_threshold 100
CMSketch *Sketch = new CMSketch(4,8192);
std::unordered_map<std::string, uint> actual_size;
int main(){
	std::string dat_path = "equinix-chicago1.dat";
	/*Insert your code here using the flowsize interface*/

	std::ifstream file(dat_path,std::ios::binary);
	if(!file){
		std::cout<<"error, file not found!\n";
		return -1;
	}
	unsigned char* buffer = new unsigned char [13];
	memset(buffer,0,13);
	uint packet_count = 0;
	int temp = 0;
	//將每個packet的data讀進來，並insert到sketch
	while(file.read(reinterpret_cast<char*>(buffer),13)|| file.gcount() > 0)
	{
		std::string data(reinterpret_cast<char*>(buffer), file.gcount());
		cuc* constData = buffer;
		// Sketch->Insert(constData);
		if(temp<10000000||temp == 10000000)
		{
			temp++;
			continue;
		}
		Sketch->Enhanced_Insert(constData);
		actual_size[data]++;
		packet_count++;
		if(packet_count == 10000000)
		{
			break;
		}
	}
	file.close();


	file.open(dat_path,std::ios::binary);
	memset(buffer,0,13);
	// FILE* out = fopen("result.txt", "w");
	FILE* counters = fopen("equinix-chicago1_counters.txt","w");
	FILE* all_flows = fopen("equinix-chicago1_flows.txt","w");
	float AAE = 0;
	float ARE = 0;
	
	//print all flows to file
	//for each flows in "actual size" print counter to file, avoid repeated print
	int number_of_flows = 0;
 	for (auto it = actual_size.begin(); it != actual_size.end(); ++it) 
	{
        cuc* temp = reinterpret_cast<cuc*>(const_cast<char*>(it->first.c_str()));
		int second = it->second;
		Sketch->Enhanced_PrintCounterFile(temp, second, all_flows);
		AAE += Sketch->CalculateAAE(temp,second);
		number_of_flows++;
    }

	printf("Total Packet Count: %u\n", packet_count);
	printf("Total Flow Count: %lu\n", actual_size.size());
	// //load parameter form pre-trained model
	// cuc* path = (unsigned char*)"parameter.txt";
	// Sketch->LoadPara(path);
	
	float aae_ml = 0;
	//query by each packet
	//這邊不用print counter，因為會重複print FLOW的counter
	while(file.read(reinterpret_cast<char*>(buffer),13)|| file.gcount() > 0)
	{	
		
		std::string data(reinterpret_cast<char*>(buffer), 13);
		cuc* constData = buffer;
		uint a = actual_size[data];
		uint query_val;
		int fearure_count = 0;
		// query_val = Sketch->Query(constData, FALSE);
		// query_val = Sketch->Enhanced_Query(constData,&fearure_count); 
		// if(query_val - a >MICE_threshold){
		// 	fprintf(out,"Actual Size: %llu Query Value: %llu feature count: %d\n", a, query_val,fearure_count);
		// 	Sketch->PrintCounterFile(constData, a, out);
		// }
		// aae_ml+=Sketch->CalculateAAE_ML(constData,a,query_val);
		Sketch->Enhanced_PrintCounterFile(constData, a, counters);
		// Sketch->PrintCounterFile(constData,actual_size[data],counters);
	}
	
	// AAE /= number_of_flows;
	//aae_ml /= packet_count because query by each packet
	// aae_ml /= packet_count;
	// std::cout<<"AAE_ML: "<<aae_ml<<std::endl;
	// std::cout<<"AAE: "<<AAE<<std::endl;
	// std::cout<<"ARE: "<<ARE<<std::endl;
	
	// fclose(out);
	fclose(counters);
	fclose(all_flows);
	return 0;
}
