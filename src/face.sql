/*
 Navicat Premium Data Transfer

 Source Server         : 39.107.42.214
 Source Server Type    : MySQL
 Source Server Version : 50647
 Source Host           : 39.107.42.214:3306
 Source Schema         : face

 Target Server Type    : MySQL
 Target Server Version : 50647
 File Encoding         : 65001

 Date: 20/03/2020 21:32:51
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for face
-- ----------------------------
DROP TABLE IF EXISTS `face`;
CREATE TABLE `face`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `face_id` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Compact;

SET FOREIGN_KEY_CHECKS = 1;
