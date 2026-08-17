#ifndef GPNN_LOGGER_H
#define GPNN_LOGGER_H

#include <string>
#include <fstream>
#include <mutex>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace gpnn {

// ============================================================================
// Logger — thread-safe logging to console + file with verbose/decision levels
// ============================================================================
// Usage:
//   Logger::init(true, "output.txt");   // verbose on, write to file
//   Logger::info("Epoch 5 started");
//   Logger::verbose("Candidate scores: IFELSE=0.82, CONTEXT_WIRE=0.75");
//   Logger::close();
//
// Normal logging (info/phase/warn): always printed to console and file.
// Verbose logging (verbose/decision): only printed when verbose mode is on.
// ============================================================================

class Logger {
public:
    // ------------------------------------------------------------------------
    // Initialization / shutdown
    // ------------------------------------------------------------------------
    static void init(bool verbose_mode, const std::string& log_file_path = "");
    static void close();

    // ------------------------------------------------------------------------
    // Normal-level logging — always shown, describes WHAT the system does
    // ------------------------------------------------------------------------
    static void info(const std::string& msg);
    static void phase(int epoch, const std::string& phase_name, const std::string& detail = "");
    static void warn(const std::string& msg);

    // ------------------------------------------------------------------------
    // Verbose-level logging — only shown in verbose mode,
    // describes WHAT DECISIONS the system makes and WHY
    // ------------------------------------------------------------------------
    static void verbose(const std::string& msg);
    static void decision(const std::string& label, const std::string& detail);

    // ------------------------------------------------------------------------
    // State queries
    // ------------------------------------------------------------------------
    static bool is_verbose() { return verbose_; }

private:
    static std::string timestamp();
    static void write(const std::string& line);
    static void write_console(const std::string& line);
    static void write_file(const std::string& line);

    static bool        verbose_;
    static bool        initialized_;
    static std::ofstream file_;
    static std::mutex  mutex_;
    static bool        use_file_;
};

} // namespace gpnn

#endif // GPNN_LOGGER_H
