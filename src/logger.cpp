#include "logger.h"
#include "constants.h"

namespace aria {

// ============================================================================
// Static members
// ============================================================================

bool        Logger::verbose_     = false;
bool        Logger::initialized_ = false;
std::ofstream Logger::file_;
std::mutex  Logger::mutex_;
bool        Logger::use_file_    = false;

// ============================================================================
// Initialization / shutdown
// ============================================================================

void Logger::init(bool verbose_mode, const std::string& log_file_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) return;

    verbose_ = verbose_mode;

    if (!log_file_path.empty()) {
        file_.open(log_file_path, std::ios::out | std::ios::app);
        if (file_.is_open()) {
            use_file_ = true;
            file_ << "\n" << config::LOG_BANNER_SEPARATOR;
            file_ << "  GP-NN Log 鈥?" << timestamp() << "\n";
            if (verbose_) file_ << "  Verbose mode enabled\n";
            file_ << config::LOG_BANNER_SEPARATOR << "\n";
        }
    }

    initialized_ = true;
}

void Logger::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) {
        file_ << "\n" << config::LOG_BANNER_SEPARATOR;
        file_ << "  Session ended 鈥?" << timestamp() << "\n";
        file_ << config::LOG_BANNER_SEPARATOR;
        file_.close();
    }
    use_file_ = false;
    initialized_ = false;
}

// ============================================================================
// Timestamp helper
// ============================================================================

std::string Logger::timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                   now.time_since_epoch()) % 1000;
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, config::LOG_TIMESTAMP_FORMAT)
        << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

// ============================================================================
// Internal write helpers
// ============================================================================

void Logger::write(const std::string& line) {
    std::lock_guard<std::mutex> lock(mutex_);
    write_console(line);
    write_file(line);
}

void Logger::write_console(const std::string& line) {
    std::cout << line << "\n";
}

void Logger::write_file(const std::string& line) {
    if (use_file_ && file_.is_open()) {
        file_ << line << "\n";
        file_.flush();  // ensure crash-safe logging
    }
}

// ============================================================================
// Normal-level logging
// ============================================================================

void Logger::info(const std::string& msg) {
    write(timestamp() + " " + config::LOG_LEVEL_INFO + " " + msg);
}

void Logger::phase(int epoch, const std::string& phase_name, const std::string& detail) {
    std::ostringstream oss;
    oss << timestamp() << " " << config::LOG_LEVEL_PHASE << " epoch=" << epoch
        << " phase=" << phase_name;
    if (!detail.empty()) oss << " " << detail;
    write(oss.str());
}

void Logger::warn(const std::string& msg) {
    write(timestamp() + " " + config::LOG_LEVEL_WARN + " " + msg);
}

// ============================================================================
// Verbose-level logging
// ============================================================================

void Logger::verbose(const std::string& msg) {
    if (!verbose_) return;
    write_console(timestamp() + " " + config::LOG_LEVEL_VERBOSE + " " + msg);
    write_file(timestamp() + " " + config::LOG_LEVEL_VERBOSE + " " + msg);
}

void Logger::decision(const std::string& label, const std::string& detail) {
    if (!verbose_) return;
    std::ostringstream oss;
    oss << timestamp() << " " << config::LOG_LEVEL_DECISION << " " << label;
    if (!detail.empty()) oss << " | " << detail;
    write_console(oss.str());
    write_file(oss.str());
}

} // namespace aria
