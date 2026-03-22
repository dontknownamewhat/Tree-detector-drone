#include "tree_detector.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

TreeDetector::TreeDetector(const std::string &record_path, const std::string &resource_root_path)
{
    m_log_root_path = record_path;
    m_resource_root_path = resource_root_path;

    std::string config_path = m_resource_root_path + "/tree_detector_config/detector.json";
    read_json_parameters(config_path);

    auto file_exists = [](const std::string &path) -> bool
    {
        std::ifstream f(path);
        return f.good();
    };

    if (!file_exists(model_path))
    {
        std::string fixed_model_path;
        if (model_path.rfind("/AI/", 0) == 0)
        {
            fixed_model_path = m_resource_root_path + model_path;
        }
        else if (!model_path.empty() && model_path[0] != '/')
        {
            fixed_model_path = m_resource_root_path + "/" + model_path;
        }

        if (!fixed_model_path.empty() && file_exists(fixed_model_path))
        {
            model_path = fixed_model_path;
        }
    }

    if (!file_exists(model_path))
    {
        throw std::runtime_error("Model file not found: " + model_path);
    }

    yolov5 = std::make_unique<YOLOv5TRT>(model_path, dla_core);
    yolov5->setConfThreshold(score_threshold);

    log("tree detector initialized");
}

void TreeDetector::setMeasureResult(std::shared_ptr<IThreadSafeData<MeasureDeviceResult>> measure_result)
{
    m_tree_result_ = measure_result;
}

void TreeDetector::setPointCloudBase(std::shared_ptr<IPointCloudBase> fw_point_cloud_base)
{
    m_fw_point_cloud_base = fw_point_cloud_base;
}

void TreeDetector::setLinearTracker(std::shared_ptr<LinearTracker> linear_tracker)
{
    m_linear_tracker_ = linear_tracker;
}

void TreeDetector::setAlgRemoteControl(IAlgRemoteControl *alg_rc)
{
    m_alg_rc_ = alg_rc;
}

void TreeDetector::setMissionStatus(std::shared_ptr<MissionStatus> mission_status)
{
    m_mission_status_ = mission_status;
}

void TreeDetector::setLogSink(std::shared_ptr<ILogSink> logger)
{
    m_logger = logger;
}

void TreeDetector::read_json_parameters(const std::string &json_file)
{
    std::ifstream ifs(json_file);
    if (!ifs.is_open())
    {
        model_path = m_resource_root_path + "/AI/tree_detector.engine";
        score_threshold = 0.5f;
        dla_core = -1;
        return;
    }

    try
    {
        json config = json::parse(ifs);

        model_path = config.value("model_path", m_resource_root_path + "/AI/tree_detector.engine");
        score_threshold = config.value("score_threshold", 0.5f);
        dla_core = config.value("dla_core", -1);

        m_x_land = config.value("x_land", 2);
        m_y_land = config.value("y_land", 2);
        m_y_large_land = config.value("y_large_land", 6);
        m_threshold = config.value("threshold", 40);
        m_min_score = config.value("min_score", 0.6f);

        m_blur_kernel_size = config.value("blur_kernel_size", 5);
        m_morph_kernel_size = config.value("morph_kernel_size", 5);
        m_min_protrusion_area = config.value("min_protrusion_area", 500);
        m_max_protrusion_count = config.value("max_protrusion_count", 10);
        m_protrusion_height_ratio = config.value("protrusion_height_ratio", 0.3f);
    }
    catch (const json::exception &)
    {
        model_path = m_resource_root_path + "/AI/tree_detector.engine";
        score_threshold = 0.5f;
        dla_core = -1;
    }
}

void TreeDetector::detect_tree(int time_sample, cv::Mat &frame_in)
{
    if (frame_in.empty())
    {
        return;
    }

    if (!m_tree_result_)
    {
        log("result buffer not set");
        return;
    }

    std::vector<Detection> detections = yolov5->detect(frame_in);

    auto tree_result = m_tree_result_->getNextData();
    if (!tree_result)
    {
        log("getNextData returned null");
        return;
    }

    tree_result->time_sample = time_sample;
    tree_result->device_type = DeviceTypeEnum::VegetationEncroachment;
    tree_result->is_detect = !detections.empty();
    tree_result->detect_list.clear();

    for (const auto &det : detections)
    {
        DeviceDetectResult tree(det.box.x, det.box.y, det.box.width, det.box.height);
        tree.score = det.conf;
        tree.dev_size = det.box.width * det.box.height;

        if (m_fw_point_cloud_base)
        {
            calculate3DPosition(tree);
        }

        tree_result->detect_list.emplace_back(tree);

        cv::Rect safe_roi = det.box & cv::Rect(0, 0, frame_in.cols, frame_in.rows);
        if (safe_roi.width <= 0 || safe_roi.height <= 0)
        {
            continue;
        }

        cv::Mat tree_roi = frame_in(safe_roi);
        std::vector<cv::Rect> protrusions = detectCrownProtrusions(tree_roi);

        if (protrusions.empty())
        {
            DeviceDetectResult fallback(tree.dev_xmin, tree.dev_ymin,
                                        tree.dev_xmax - tree.dev_xmin,
                                        tree.dev_ymax - tree.dev_ymin);
            fallback.score = tree.score;
            fallback.dev_size = tree.dev_size;
            fallback.dx = tree.dx;
            fallback.dy = tree.dy;
            fallback.dz = tree.dz;

            if (fallback.dz > 0.0f)
            {
                calculateClearanceDistance(fallback);
            }

            tree_result->detect_list.emplace_back(fallback);
        }

        for (const auto &part_rect : protrusions)
        {
            int global_x = safe_roi.x + part_rect.x;
            int global_y = safe_roi.y + part_rect.y;

            DeviceDetectResult part(global_x, global_y, part_rect.width, part_rect.height);
            part.score = tree.score;
            part.dev_size = part_rect.width * part_rect.height;

            bool has_depth = false;
            if (m_fw_point_cloud_base)
            {
                has_depth = calculate3DPosition(part);
                if (!has_depth && tree.dz > 0.0f)
                {
                    part.dx = tree.dx;
                    part.dy = tree.dy;
                    part.dz = tree.dz;
                    has_depth = true;
                }

                if (has_depth)
                {
                    calculateClearanceDistance(part);
                }
            }

            tree_result->detect_list.emplace_back(part);
        }
    }

    if (tree_result->detect_list.size() > 1)
    {
        std::sort(tree_result->detect_list.begin(),
                  tree_result->detect_list.end(),
                  [](const DeviceDetectResult &a, const DeviceDetectResult &b)
                  {
                      return a.dev_size > b.dev_size;
                  });
    }

    // 脱敏后仅保留通用告警回调，不包含设备ID、数据库、照片路径等业务细节。
    bool has_clearance_point = false;
    float min_tree_dz = FLT_MAX;
    float min_clearance = FLT_MAX;

    for (const auto &item : tree_result->detect_list)
    {
        if (item.has_clearance_point)
        {
            has_clearance_point = true;
            min_clearance = std::min(min_clearance, item.dxmax);
        }
        if (item.dz > 0.01f)
        {
            min_tree_dz = std::min(min_tree_dz, item.dz);
        }
    }

    bool within_record_distance = (min_tree_dz < 6.5f);
    bool cooldown_ok = true;
    auto now = std::chrono::steady_clock::now();
    if (m_record_time_valid)
    {
        cooldown_ok = (now - m_last_record_time) >= m_record_cooldown;
    }

    if (m_alg_rc_ && has_clearance_point && within_record_distance && cooldown_ok && !tree_result->detect_list.empty())
    {
        const auto &target = tree_result->detect_list.front();
        cv::Rect roi(target.dev_xmin,
                     target.dev_ymin,
                     target.dev_xmax - target.dev_xmin,
                     target.dev_ymax - target.dev_ymin);

        m_alg_rc_->onTreeAlert(time_sample, min_tree_dz, min_clearance, roi);
        m_last_record_time = now;
        m_record_time_valid = true;
    }

    m_tree_result_->changeToNextData();
}

bool TreeDetector::calculate3DPosition(DeviceDetectResult &result)
{
    if (!m_fw_point_cloud_base)
    {
        return false;
    }

    auto measure_result = m_fw_point_cloud_base->getMeasureResult();
    if (!measure_result)
    {
        return false;
    }

    auto current_data = measure_result->getCurrentData();
    if (!current_data)
    {
        return false;
    }

    cv::Mat left_gray = current_data->imageLeft;
    cv::Mat right_gray = current_data->imageRight;

    PointCloudDetectConfig config = current_data->config;
    config.detect_start_x = result.dev_xmin;
    config.detect_start_y = result.dev_ymin;
    config.detect_end_x = result.dev_xmax;
    config.detect_end_y = result.dev_ymax;
    config.x_land = m_x_land;
    config.y_land = m_y_land;
    config.y_large_land = m_y_large_land;
    config.threshold = m_threshold;
    config.min_score = m_min_score;

    std::vector<PointCloudData> point_list;
    m_fw_point_cloud_base->calculatePointCloudForList(left_gray, right_gray, config, point_list);
    if (point_list.empty())
    {
        return false;
    }

    std::sort(point_list.begin(), point_list.end(),
              [](const PointCloudData &a, const PointCloudData &b)
              {
                  return a.dz < b.dz;
              });

    result.dxmin = FLT_MAX;
    result.dxmax = FLT_MIN;
    result.dymin = FLT_MAX;
    result.dymax = FLT_MIN;

    int pre_count = 3;
    int pre_dz = -1;
    float avg_count = 0.0f;
    bool valid = false;

    for (size_t i = 0; i < point_list.size() && pre_count != 0; ++i)
    {
        PointCloudData &pt = point_list[i];

        if (static_cast<int>(pt.dz) != pre_dz)
        {
            pre_dz = static_cast<int>(pt.dz);
            pre_count--;
        }

        result.dx = ((result.dx * avg_count) + pt.dx) / (avg_count + 1.0f);
        result.dy = ((result.dy * avg_count) + pt.dy) / (avg_count + 1.0f);
        result.dz = ((result.dz * avg_count) + pt.dz) / (avg_count + 1.0f);

        result.dxmin = std::min(result.dxmin, pt.dx);
        result.dxmax = std::max(result.dxmax, pt.dx);
        result.dymin = std::min(result.dymin, pt.dy);
        result.dymax = std::max(result.dymax, pt.dy);

        avg_count += 1.0f;
        valid = true;
    }

    return valid;
}

void TreeDetector::calculateClearanceDistance(DeviceDetectResult &tree_result)
{
    if (!m_fw_point_cloud_base || tree_result.dz <= 0.0f)
    {
        return;
    }

    auto measure_result = m_fw_point_cloud_base->getMeasureResult();
    if (!measure_result)
    {
        return;
    }

    auto current_data = measure_result->getCurrentData();
    if (!current_data)
    {
        return;
    }

    cv::Mat left_gray = current_data->imageLeft;
    cv::Mat right_gray = current_data->imageRight;

    PointCloudDetectConfig config = current_data->config;
    config.detect_start_x = 0;
    config.detect_start_y = 0;
    config.detect_end_x = left_gray.cols;
    config.detect_end_y = left_gray.rows;
    config.x_land = m_x_land;
    config.y_land = m_y_land;
    config.y_large_land = m_y_large_land;
    config.threshold = m_threshold;
    config.min_score = m_min_score;

    std::vector<PointCloudData> point_list;
    m_fw_point_cloud_base->calculatePointCloudForList(left_gray, right_gray, config, point_list);
    if (point_list.empty())
    {
        return;
    }

    const float depth_tol = 1.0f;
    const float height_tol = 10.0f;
    const float horiz_center = 3.0f;
    const float horiz_tol = 2.0f;

    float min_clearance = FLT_MAX;
    PointCloudData best_point;
    int match_count = 0;

    for (const auto &pt : point_list)
    {
        if (std::abs(pt.dz - tree_result.dz) > depth_tol)
            continue;
        if (std::abs(pt.dx - tree_result.dx) > height_tol)
            continue;
        if (std::abs(std::abs(pt.dy) - horiz_center) > horiz_tol)
            continue;

        float ddx = pt.dx - tree_result.dx;
        float ddy = pt.dy - tree_result.dy;
        float ddz = pt.dz - tree_result.dz;
        float dist = std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);

        if (dist < min_clearance)
        {
            min_clearance = dist;
            best_point = pt;
        }
        match_count++;
    }

    if (match_count == 0 || min_clearance == FLT_MAX)
    {
        return;
    }

    tree_result.dxmin = best_point.dx - tree_result.dx;
    tree_result.dymin = best_point.dy - tree_result.dy;
    tree_result.dxmax = min_clearance;

    int px = static_cast<int>(std::lround(best_point.piexl_x));
    int py = static_cast<int>(std::lround(best_point.piexl_y));
    if (!left_gray.empty())
    {
        px = std::max(0, std::min(px, left_gray.cols - 1));
        py = std::max(0, std::min(py, left_gray.rows - 1));
    }

    tree_result.clearance_px = px;
    tree_result.clearance_py = py;
    tree_result.has_clearance_point = true;
}

void TreeDetector::drawDetectionResult(cv::Mat &frame, const MeasureDeviceResult &result)
{
    if (!result.is_detect || result.detect_list.empty())
    {
        return;
    }

    for (const auto &tree : result.detect_list)
    {
        cv::rectangle(frame,
                      cv::Point(tree.dev_xmin, tree.dev_ymin),
                      cv::Point(tree.dev_xmax, tree.dev_ymax),
                      cv::Scalar(0, 255, 0),
                      2);

        int x_center = (tree.dev_xmin + tree.dev_xmax) / 2;
        int y_center = (tree.dev_ymin + tree.dev_ymax) / 2;
        cv::circle(frame, cv::Point(x_center, y_center), 8, cv::Scalar(0, 0, 255), -1);

        if (tree.has_clearance_point)
        {
            cv::circle(frame, cv::Point(tree.clearance_px, tree.clearance_py), 6, cv::Scalar(0, 255, 255), -1);
        }

        std::string info;
        char buffer[128];

        if (tree.dxmax > 0.01f)
        {
            snprintf(buffer, sizeof(buffer), "Clr: %.2fm", tree.dxmax);
            info = buffer;
            if (tree.dz > 0.01f)
            {
                snprintf(buffer, sizeof(buffer), " (Tree: %.2fm)", tree.dz);
                info += buffer;
            }
        }
        else if (tree.dz > 0.01f)
        {
            snprintf(buffer, sizeof(buffer), "Tree: %.2fm", tree.dz);
            info = buffer;
        }
        else
        {
            info = "Tree";
        }

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(info, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        int text_x = std::max(tree.dev_xmin, 0);
        int text_y = std::max(tree.dev_ymin - 10, text_size.height + 5);

        cv::rectangle(frame,
                      cv::Point(text_x, text_y - text_size.height - 5),
                      cv::Point(text_x + text_size.width, text_y + baseline),
                      cv::Scalar(0, 255, 255),
                      -1);
        cv::putText(frame,
                    info,
                    cv::Point(text_x, text_y),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(0, 0, 0),
                    2);
    }
}

std::vector<cv::Rect> TreeDetector::detectCrownProtrusions(const cv::Mat &tree_roi)
{
    std::vector<cv::Rect> protrusions;
    if (tree_roi.empty())
    {
        return protrusions;
    }

    cv::Mat gray;
    if (tree_roi.channels() == 3)
    {
        cv::cvtColor(tree_roi, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = tree_roi.clone();
    }

    cv::Mat blurred;
    int kernel_size = m_blur_kernel_size;
    if (kernel_size % 2 == 0)
    {
        kernel_size++;
    }
    cv::GaussianBlur(gray, blurred, cv::Size(kernel_size, kernel_size), 0);

    cv::Mat edges;
    cv::Canny(blurred, edges, 50, 150);

    cv::Mat morph_edges;
    int morph_size = std::max(1, m_morph_kernel_size);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morph_size, morph_size));
    cv::morphologyEx(edges, morph_edges, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(morph_edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int tree_height = tree_roi.rows;
    float height_threshold = tree_height * m_protrusion_height_ratio;

    std::vector<std::pair<cv::Rect, double>> candidates;
    for (const auto &contour : contours)
    {
        cv::Rect bbox = cv::boundingRect(contour);
        double area = cv::contourArea(contour);

        if (area >= m_min_protrusion_area &&
            bbox.y < height_threshold &&
            bbox.height < tree_height * 0.5 &&
            bbox.width < tree_roi.cols * 0.8)
        {
            candidates.push_back({bbox, area});
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const std::pair<cv::Rect, double> &a, const std::pair<cv::Rect, double> &b)
              {
                  return a.second > b.second;
              });

    int max_count = std::min(static_cast<int>(candidates.size()), m_max_protrusion_count);
    for (int i = 0; i < max_count; ++i)
    {
        protrusions.push_back(candidates[i].first);
    }

    return protrusions;
}

void TreeDetector::log(const std::string &msg) const
{
    if (m_logger)
    {
        m_logger->log(msg);
    }
}
