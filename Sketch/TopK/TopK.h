#ifndef TOPK_H
#define TOPK_H

// #include "Sketch.h"
#include "CUSketch.h"
// #include "CMSketch.h"
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <iostream>

typedef const unsigned char cuc;

// Flow counter with old (initial) and new (current) value
struct NO {
	uint oldValue;
	uint newValue;

	NO(uint o, uint n) : oldValue(o), newValue(n) {}
	NO(uint n) : oldValue(n), newValue(n) {}

	friend uint operator++(NO &no) {
		return ++no.newValue;
	}
	friend uint operator++(NO &no, int) {
		return no.newValue++;
	}
	friend bool operator<(const NO &no, uint num) {
		return no.newValue < num;
	}
	operator uint() const {
		return newValue;
	}
};

class TopK {
public:
	TopK(uint type, uint d, uint w, uint k);
	~TopK();

	// 外部設定訓練環境
	FILE* training_file;
	std::unordered_set<std::string>* actual_topk_set;
	void SetEnv(FILE* out, std::unordered_set<std::string>* label_set);

	// 工具函式：binary string → hex 字串
	void Insert(cuc* str);
	void PrintTopK();
	std::vector<std::pair<std::string, uint>> GetTopK();
	void DumpToCSV(const std::string& filename);

private:
	CUSketch* sketch;
	uint k;
	std::unordered_map<std::string, NO>* heap;
	std::unordered_map<std::string, NO>::iterator MP; // my_min pointer
	uint my_min;
};


#endif // TOPK_H