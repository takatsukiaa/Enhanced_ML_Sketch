#ifndef CMSKETCH_H
#define CMSKETCH_H
#include <climits>
#include "common.h"
#define THRESH 28
struct CMSketch:public Sketch{
public:
	CMSketch(uint d, uint w);
	~CMSketch();
	void Insert(cuc *str);
	void Enhanced_Insert(cuc* str);
	void GetHashedValue(cuc *str, uint *counters);
	ull Query(cuc *str, bool ml = FALSE);
	void PrintCounter(cuc* str, uint acc_val);
	void PrintCounterFile(cuc * str, uint acc_val, FILE * fout);
	float CalculateAAE(cuc * str, uint acc_val);
	void LoadPara(cuc *path = CMPATH);
	float Predict(ull *t);
	float CalculateAAE_ML(cuc * str, uint acc_val, float query_val);
	float CalculateARE(cuc * str, uint acc_val);
	ull Enhanced_Query(cuc* str,int* feature_count);
	void Enhanced_PrintCounterFile(cuc * str, uint acc_val, FILE * fout);
private:
	HashFunction *hf;
	ull** sketch;
	float* para;
	float* mean;
	float* scale;
	uint d;
	uint w;
	ull *t;
	uchar** ov_flags;
};


CMSketch::CMSketch(uint d, uint w):d(d), w(w){
	sketch = new ull*[d];
	ov_flags = new uchar*[d];
	for(uint i = 0; i < d; ++i){
		sketch[i] = new ull[w];
		ov_flags[i] = new uchar[w];
		memset(sketch[i], 0, w * sizeof(ull));
		memset(ov_flags[i], 0, w * sizeof(uchar));
	}
	hf = new HashFunction();
	para = new float[d];
	mean = new float[d];
	scale = new float[d];
	t = new ull[d];
}

CMSketch::~CMSketch(){
	for(uint i = 0; i < d; ++i) delete [] sketch[i];
	delete [] sketch;
	delete hf;
	delete [] para;
	delete [] mean;
	delete [] scale;
	delete [] t;
}

void CMSketch::Insert(cuc *str){
    for (uint i = 0; i < d; ++i){
        uint cid = hf->Str2Int(str, i) % w;
        if (sketch[i][cid] == -1) {
            return;
        }
		++sketch[i][cid];
	}
}
void CMSketch::Enhanced_Insert(cuc* str)
{
	ull min = 0;
	uint cid[3];
	for(uint i=0; i < d; i++)
	{
		cid[i] = hf->Str2Int(str, i) % w;
		if(sketch[i][cid[i]] == -1)
		{
			return;
		}
	}
	
	for(uint i = 0; i < d; i++)
	{
		if(ov_flags[i][cid[i]] == '\0')
		{
			sketch[i][cid[i]] = sketch[i][cid[i]]<<32>>32;
		}
		++sketch[i][cid[i]];
		if(i == 0)
		{
			min = sketch[i][cid[i]];
		}
		if(sketch[i][cid[i]] < min || sketch[i][cid[i]] == min)
		{
			min = sketch[i][cid[i]];
		}
		if(sketch[i][cid[i]]>UINT_MAX)
		{
			ov_flags[i][cid[i]] = 1;
		}
	}
	for(uint i = 0; i < d; i++)
	{
		if(ov_flags[i][cid[i]] == '\0')
		{
			sketch[i][cid[i]] += min<<32;
		}
	}
}
ull CMSketch::Enhanced_Query(cuc* str, int* feature_count)
{
    ull min = ULLONG_MAX;
    uint cid[3];
    
    // Step 1: 計算每行的哈希索引
    for (uint i = 0; i < d; i++) {
        cid[i] = hf->Str2Int(str, i) % w;
    }

    // Step 2: 遍歷每一行，獲取計數器值
    for (uint i = 0; i < d; i++) {
        ull value;
        
        // 檢查溢出標誌
        if (ov_flags[i][cid[i]] == 1) {
            // 如果溢出，取整個 64 位值
            value = sketch[i][cid[i]];
			*feature_count +=1;
        } else {
            // 如果未溢出，僅取後 32 位值
            value = sketch[i][cid[i]] & 0xFFFFFFFF;
			*feature_count+=2;
        }
        
        // 更新最小值
        if (value < min) {
            min = value;
        }
    }

    // Step 3: 返回最小值
    return min;
}

void CMSketch::GetHashedValue(cuc *str, uint *counters)
{
	for(uint i = 0; i < d; i++)
	{
		counters[i] = hf->Str2Int(str,i) % w;
	}
}

ull CMSketch::Query(cuc *str, bool ml){
	memset(t, 0, sizeof(t)*3);

	ull Min;
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	std::sort(t,t+d);
	if (!ml || !need_analyze(t, d)) {
		Min = t[0];
		return Min;
	}
	else {
		printf("using ml:\n");
		std::sort(t, t+d);
		ull result = Predict(t);
		//if you want a float;
		if(t[3] - t[1] <= THRESH && result>0 && result <= t[0])
		{
			return result;
		}
		return t[0];
		// if(result > t[0] || result <= 0)
		// {
		// 	return t[0];
		// }
		// else
		// {
		// 	return result;
		// }
		//if you want a integer;
		//return (uint)Predict(t);
	}
}

void CMSketch::PrintCounter(cuc* str, uint acc_val){
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	if (need_analyze(t, d)) {
		std::sort(t, t + d);
		printf("Actual Size: ");
		printf("%u", acc_val);
		printf("Counter Values:");
		for(uint i = 0; i < d; ++i){
			printf(" %llu", t[i]);
		}
		printf("\n");
	}
}
ull CMSketch::Enhanced_Query(cuc* str, int* feature_count)
{
    ull min = ULLONG_MAX;
    uint cid[3];
    
    // Step 1: 計算hash值
    for (uint i = 0; i < d; i++) {
        cid[i] = hf->Str2Int(str, i) % w;
    }

    // Step 2: 遍歷每一行，獲取計數器值
    for (uint i = 0; i < d; i++) {
        ull value;
        
        // 檢查溢出標誌
        if (ov_flags[i][cid[i]] == 1) {
            // 如果溢出，取整個 64 位值
            value = sketch[i][cid[i]];
			*feature_count +=1;
        } else {
            // 如果未溢出，僅取後 32 位值
            value = sketch[i][cid[i]] & 0xFFFFFFFF;
			*feature_count+=2;
        }
        
        // 更新最小值
        if (value < min) {
            min = value;
        }
    }

    // Step 3: 返回最小值
    return min;
}
void CMSketch::Enhanced_PrintCounterFile(cuc* str, uint acc_val, FILE* fout) {
    uint cid[3];
    ull value;
    int feature_count = 0;
    //計算 Hash 值
    for (uint i = 0; i < d; i++) {
        cid[i] = hf->Str2Int(str, i) % w;
    }
	// 計算特徵數
    for (uint i = 0; i < d; i++) {
        if (ov_flags[i][cid[i]] == 1) {
           feature_count += 1;
        } else {
           feature_count += 2;
        }
    }
	fprintf(fout, "%d", feature_count);
    //寫入檔案
    fprintf(fout, " %u", acc_val); // 實際值
    for (uint i = 0; i < d; i++) {
        if (ov_flags[i][cid[i]] == 1) {
            // 如果溢出，輸出 64 位值
            value = sketch[i][cid[i]];
            fprintf(fout, " %llu", value);
        } else {
            // 如果未溢出，輸出兩個 32 位值
            uint low = sketch[i][cid[i]] & 0xFFFFFFFF;
            uint high = (sketch[i][cid[i]] >> 32) & 0xFFFFFFFF;
            fprintf(fout, " %u %u", low, high);
        }
    }
    fprintf(fout, "\n"); 
}


void CMSketch::PrintCounterFile(cuc * str, uint acc_val, FILE * fout)
{
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	std::sort(t, t + d);
	// fprintf(fout, "Actual Size: ");
	fprintf(fout, "%u ", acc_val);
	// fprintf(fout, "Counter Values:");
	for(uint i = 0; i < d-1; ++i){
		fprintf(fout, "%llu ", t[i]);
	}
	fprintf(fout,"%llu",t[d-1]);
	fprintf(fout, "\n");
}

void CMSketch::LoadPara(cuc *path){
	float temp;
	FILE *file = fopen((const char*)path, "r");
	for(uint i = 0; i < d; ++i){
		fscanf(file, "%f", mean+i);
	}
	fscanf(file, "%f", &temp);
	for(uint i = 0; i < d; ++i){
		fscanf(file, "%f", scale+i);
	}
	fscanf(file, "%f", &temp);
	for(uint i = 0; i < d; ++i){
		fscanf(file, "%f", para+i);
	}
}
float CMSketch::CalculateAAE(cuc * str, uint acc_val)
{
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	std::sort(t, t + d);
	return abs((float)t[0] - acc_val);
}
float CMSketch::CalculateAAE_ML(cuc * str, uint acc_val, float query_val)
{
	return abs(query_val - acc_val);
}
float CMSketch::CalculateARE(cuc* str, uint acc_val)
{
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	std::sort(t, t + d);
	return abs(((float)t[0] - acc_val)/acc_val);
}
float CMSketch::Predict(ull *t){
	float res = 0;
	for(uint i = 0; i < d; ++i){
		res += para[i]*((t[i]-mean[i])/scale[i]);
	}
	return res;
}

#endif