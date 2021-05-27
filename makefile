EVA_PATH        = ../eva
EVA_LIB         = ${EVA_PATH}/.build_release/libeva.a

MLVM_PATH       = ../mlvm

include ${EVA_PATH}/eva.mk

# ------------------------------------------------------------------------------
# Configurations.
# ------------------------------------------------------------------------------

SRC             =  src
CMD             =  cmd
FMT_FOLDERS     =  ${SRC} ${CMD}  # required by eva.mk

CFLAGS          += -I${SRC} -I${EVA_PATH}/src -I${MLVM_PATH}/src -g
LDFLAGS         += ${EVA_LIB}

# ------------------------------------------------------------------------------
# Libs.
# ------------------------------------------------------------------------------
BB_HEADER       = ${SRC}/bb.h
BB_LIB          = ${BUILD}/bb_bb.o

ALL_LIBS        = ${BB_LIB}

# ------------------------------------------------------------------------------
# Actions.
# ------------------------------------------------------------------------------

.DEFAULT_GOAL   = compile

compile: ${BUILD} ${ALL_LIBS}

${BUILD}/bb_%.o: ${SRC}/%.c ${BB_HEADER}
	${EVA_CC} -o $@ -c $<

# ------------------------------------------------------------------------------
# Cmd.
# ------------------------------------------------------------------------------

# Put `test` out from CMDS, as it needs special testing library in next section.
CMD_CANDIDATES  = $(patsubst ${CMD}/%,%,$(wildcard ${CMD}/*))
CMDS            = $(filter-out test,${CMD_CANDIDATES})
CMD_TARGETS     = $(patsubst ${CMD}/%/main.c,${BUILD}/%,$(wildcard ${CMD}/*/main.c))

compile: ${CMD_TARGETS}

$(foreach cmd,$(CMDS),$(eval $(call objs,$(cmd),$(BUILD),$(BB_LIB))))

