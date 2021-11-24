EVA_PATH        = ../eva
EVA_LIB         = ${EVA_PATH}/.build_release/libeva.a

MLVM_PATH       = ../mlvm
MLVM_LIB        = ${MLVM_PATH}/.build_release/libmlvm.a

# The template eva.mk offers BUILD (build dir var), MK (action var for different
# platforms), fmt (action), objs (make fn), etc.
include ${EVA_PATH}/eva.mk

# ------------------------------------------------------------------------------
# configurations.
# ------------------------------------------------------------------------------

SRC             =  src
CMD             =  cmd
FMT_FOLDERS     =  ${SRC} ${CMD}  # required by eva.mk


CFLAGS          += -I${SRC} -I${EVA_PATH}/src
CFLAGS          += -DVM_SPEC -I${MLVM_PATH}/src -I${MLVM_PATH}/include
LDFLAGS         += ${MLVM_LIB} ${EVA_LIB}

LDFLAGS         += -lncurses

# ------------------------------------------------------------------------------
# libs.
# ------------------------------------------------------------------------------

ALL_LIBS         = ${BUILD}/bb_bot.o ${BUILD}/bb_board.o ${BUILD}/bb_runner.o

# ------------------------------------------------------------------------------
# actions.
# ------------------------------------------------------------------------------

.DEFAULT_GOAL   = compile

compile: ${BUILD} ${ALL_LIBS}

${BUILD}/bb_%.o: ${SRC}/%.c
	${EVA_CC} -o $@ -c $<

# ------------------------------------------------------------------------------
# cmds.
# ------------------------------------------------------------------------------

# filter `test` out from CMDS, as it needs special testing library.
CMD_CANDIDATES  = $(patsubst ${CMD}/%,%,$(wildcard ${CMD}/*))
CMDS            = $(filter-out test,${CMD_CANDIDATES})
CMD_TARGETS     = $(patsubst ${CMD}/%/main.c,${BUILD}/%,$(wildcard ${CMD}/*/main.c))

compile: ${CMD_TARGETS}

$(foreach cmd,$(CMDS),$(eval $(call objs,$(cmd),$(BUILD),$(ALL_LIBS))))

# ------------------------------------------------------------------------------
# deps.
# ------------------------------------------------------------------------------

# start with a fresh build (-B).
DEP_FLAGS      += -B -j

ifdef RELEASE
DEP_FLAGS      += RELEASE=1
endif

ifdef BLIS
DEP_FLAGS      += BLIS=1
endif

ifdef RELEASE
deps:
	${MK} -C ../eva ${DEP_FLAGS} libeva
	${MK} -C ../mlvm ${DEP_FLAGS} libmlvm
else
deps:
	@echo "[g]make deps must be in RELEASE mode."
endif

