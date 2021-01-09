#pragma once

class Options;

class InputParser
{
public:
	static void parse(Options* options, int argc, char* argv[]);
};